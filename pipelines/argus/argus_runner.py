import gc

import torch


class ArgusRunner(torch.nn.Module):
    """Runs Argus 360° out-painting over a video in overlapping windows and blends the latents."""

    def __init__(self, accelerator, weight_dtype, argus_config, argus_pipeline, device=torch.device("cuda:0")):
        super().__init__()
        self.device = device
        self.accelerator = accelerator
        self.weight_dtype = weight_dtype
        self.argus_config = argus_config
        self.argus_pipeline = argus_pipeline

    @torch.no_grad()
    def forward_argus(self, input_video, eq_video, eq_mask, noise_conditioning=False, return_unet_latents=True,
                      offload_to_cpu=True):
        conditional_video_pers = input_video
        conditional_video_equi = eq_video.to(self.weight_dtype)
        mask = eq_mask

        if noise_conditioning:
            conditional_video_equi = torch.where(mask == 1, input_video,
                                                 torch.randn_like(input_video, device=self.accelerator.device))
        conditional_video_equi = conditional_video_equi.unsqueeze(0)
        hw_ratio = conditional_video_pers.shape[-2] / conditional_video_pers.shape[-1]
        mask = mask.unsqueeze(0)
        conditional_video_pers = conditional_video_pers.unsqueeze(0)

        with torch.autocast(str(self.accelerator.device).replace(":0", ""),
                            enabled=self.accelerator.mixed_precision != 'no',
                            dtype=self.weight_dtype):
            num_frames_remaining = self.argus_config.num_frames
            num_frames_batch = self.argus_config.num_frames_batch if (
                    hasattr(self.argus_config,
                            'num_frames_batch') and self.argus_config.num_frames_batch is not None) else self.argus_config.num_frames
            generated_frames_this = None
            all_extracted_latents = None
            num_frames_processed = 0
            blend_frames = self.argus_config.blend_frames if hasattr(self.argus_config, 'blend_frames') else 0
            round = 0

            all_generated_latents = None

            while num_frames_remaining > 0:
                num_frames_to_process = min(num_frames_batch, num_frames_remaining)
                if round == 0:
                    conditional_video_input = conditional_video_equi[:, :num_frames_to_process]
                else:
                    conditional_video_input = torch.cat(
                        [generated_frames_this[:, :, -blend_frames:].permute(0, 2, 1, 3, 4),
                         conditional_video_equi[
                             :, num_frames_processed + blend_frames: num_frames_processed + num_frames_to_process]],
                        dim=1)
                conditional_video_input = conditional_video_input

                mask_this = mask[:, num_frames_processed: num_frames_processed + num_frames_to_process]
                conditional_video_pers_this = conditional_video_pers[
                    :, num_frames_processed: num_frames_processed + num_frames_to_process]
                # Pin the RNG for noise augmentation AND the Argus initial latent so
                # the generated panorama is bit-identical across runs regardless of
                # what torch kernels consumed RNG state earlier in the pipeline.
                _argus_gen = torch.Generator(device=conditional_video_input.device).manual_seed(0)
                conditional_video_input = conditional_video_input + torch.randn(
                    conditional_video_input.shape, generator=_argus_gen,
                    device=conditional_video_input.device, dtype=conditional_video_input.dtype,
                ) * self.argus_config.noise_aug_strength * mask_this

                fps = 3.0  # Argus conditioning frame rate

                conditional_video_input = conditional_video_input.to(self.weight_dtype).to(self.accelerator.device)
                conditional_video_pers_this = conditional_video_pers_this.to(self.weight_dtype).to(self.accelerator.device)
                generated_latents_this, extracted_latents_this = self.argus_pipeline(
                    conditional_video_input,  # (1, T, C, H, W)
                    conditional_images=conditional_video_pers_this,  # (1, T, C, H, W)
                    height=self.argus_config.height,
                    width=self.argus_config.width,
                    num_frames=num_frames_to_process,
                    decode_chunk_size=self.argus_config.decode_chunk_size,
                    motion_bucket_id=127,
                    fps=fps,
                    num_inference_steps=self.argus_config.num_inference_steps,
                    noise_aug_strength=self.argus_config.noise_aug_strength,
                    min_guidance_scale=self.argus_config.guidance_scale,
                    max_guidance_scale=self.argus_config.guidance_scale,
                    inference_final_rotation=self.argus_config.inference_final_rotation,
                    blend_decoding_ratio=self.argus_config.blend_decoding_ratio,
                    extended_decoding=self.argus_config.extended_decoding,
                    rotation_during_inference=self.argus_config.rotation_during_inference,
                    return_latents=True,
                    get_final_unet_latents=return_unet_latents,
                    # Pin Argus's initial latent so the panorama is reproducible run-to-run.
                    generator=torch.Generator(device=conditional_video_input.device).manual_seed(0),
                )  # [B, T, C, H, W]
                conditional_video_input = conditional_video_input.cpu()
                conditional_video_pers_this = conditional_video_pers_this.cpu()
                del conditional_video_input
                del mask_this
                del conditional_video_pers_this
                torch.cuda.empty_cache()

                generated_frames_this = self.argus_pipeline.decode_latents(generated_latents_this,
                                                                           num_frames_to_process,
                                                                           decode_chunk_size=self.argus_config.decode_chunk_size,
                                                                           extended_decoding=self.argus_config.extended_decoding,
                                                                           blend_decoding_ratio=self.argus_config.blend_decoding_ratio)
                if round == 0:
                    B, _, C, H, W = generated_latents_this.shape
                    all_generated_latents = torch.empty((B, conditional_video_equi.shape[1], C, H, W),
                                                        dtype=self.weight_dtype,
                                                        device=torch.device(
                                                            "cpu") if offload_to_cpu else self.accelerator.device)
                    generated_latents_this = generated_latents_this.cpu()

                    all_generated_latents[:, :num_frames_to_process, ...] = generated_latents_this.to(
                        all_generated_latents.device)
                    all_extracted_latents = extracted_latents_this
                else:
                    generated_latents_this = generated_latents_this.cpu()
                    start_idx = num_frames_processed
                    blend_start = start_idx
                    blend_end = start_idx + blend_frames
                    new_data_end = start_idx + num_frames_to_process

                    blend_weight = torch.linspace(1, 0, blend_frames,
                                                  device=all_generated_latents.device,
                                                  dtype=all_generated_latents.dtype).view(1, blend_frames, 1, 1, 1)

                    existing_data = all_generated_latents[:, blend_start:blend_end, ...]
                    all_generated_latents[:, blend_start:blend_end, ...] = blend_weight * existing_data + (
                            1 - blend_weight) * generated_latents_this[:, :blend_frames, ...]

                    all_generated_latents[:, blend_end:new_data_end, ...] = generated_latents_this[
                        :, blend_frames:, ...]
                    # Extracted UNet latents are only blended for batch size 1.
                    if extracted_latents_this is not None:
                        for k1 in all_extracted_latents.keys():
                            for k2 in all_extracted_latents[k1].keys():
                                all_extracted_latents[k1][k2] = torch.cat(
                                    [all_extracted_latents[k1][k2][:-blend_frames],
                                     blend_weight[0] * all_extracted_latents[k1][k2][-blend_frames:] + (
                                             1 - blend_weight[0]) * extracted_latents_this[k1][k2][:blend_frames],
                                     extracted_latents_this[k1][k2][blend_frames:]], dim=0)

                if num_frames_remaining == num_frames_to_process:
                    break

                generated_frames_this = generated_frames_this.to(all_generated_latents.device)

                num_frames_remaining -= (num_frames_to_process - blend_frames)
                num_frames_processed += (num_frames_to_process - blend_frames)
                round += 1
                del generated_latents_this, extracted_latents_this
                gc.collect()
                torch.cuda.empty_cache()
                self.accelerator.free_memory()

        del conditional_video_pers
        return all_generated_latents, all_extracted_latents

import gc
import json
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator

from conversions.mapper import Mappers
from metrics.tapvid360_metrics import compute_metrics
from metrics.motion_aware_metrics import compute_metrics_by_motion_type
from runners.settings import get_args, load_config
from runners.model_utils import SAMRunner, get_models, get_360_latents_model
from runners.pipeline_helpers import _decode_pred_eq_frames, _resize_eq_frames, run_through_cotracker
from runners.utils import overlay_orig_persp_on_pred_eq, get_dataset, get_object_centred_persp_imgs_with_interpolation
from runners.vis_utils import overlay_mask_over_image
from runners.pose_utils import VGGTPoseRunner

def _save_eq_frames(eq_frames_torch, out_dir):
    """Save per-frame equirectangular panoramas as JPEGs (uint8 RGB)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = eq_frames_torch.detach().cpu().to(torch.uint8).numpy()
    for i, frame in enumerate(arr):
        Image.fromarray(frame).save(out_dir / f"{i}.jpg")


def _save_sam_overlays(pred_eq_frames_torch, seg_masks, out_dir):
    """Save per-frame SAM2 mask overlays on top of the Argus panorama."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = pred_eq_frames_torch.detach().cpu().to(torch.uint8).numpy()
    masks = seg_masks.detach().cpu().numpy()
    for i in range(len(frames)):
        overlay = overlay_mask_over_image(frames[i], masks[i])
        Image.fromarray(overlay).save(out_dir / f"{i}.jpg")


def _save_obj_centre_frames(obj_cnt_persp_imgs, out_dir):
    """Save per-frame object-centred perspective crops."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = obj_cnt_persp_imgs.detach().cpu().to(torch.uint8).numpy()
    for i, frame in enumerate(arr):
        Image.fromarray(frame).save(out_dir / f"{i}.jpg")


def get_rotations(data, device, use_gt=False, offload_from_gpu=False):
    fov_x = data.fov_x[0]
    if use_gt:
        R_w2c = data.rotations[0]
    else:
        pose_runner = VGGTPoseRunner(device=device)
        # For our definition of predicted rotations we can use the original ground truth fov_x and starting rotation position
        pred_extrinsics, _ = pose_runner.run(data.video[0].permute(0, 2, 3, 1))
        pred_rots = pose_runner.convert_to_tapvid360_format(pred_extrinsics[..., :3]).to(torch.float32)[0]
        R_w2c = pose_runner.align_to_ground_truth_start_rotations(pred_rots, data.rotations[0, 0])
        if offload_from_gpu:
            pose_runner.model.cpu()
            del pose_runner
            torch.cuda.empty_cache()

    return R_w2c.to(device), fov_x

def create_argus_inputs(persp_video_frames, rotations, mapper):
    conditional_video_equi, mask = mapper.image.perspective_image_to_equirectangular(persp_video_frames, rotations,
                                                                                     to_uint=False)
    conditional_video_equi = conditional_video_equi.clip(0, 1)
    # Put in the range of [-1, 1] for Argus
    conditional_video_equi = conditional_video_equi * 2 - 1
    conditional_video_equi = conditional_video_equi.permute(0, 3, 1, 2)
    mask = mask.unsqueeze(1)
    # Argus needs of shape 512, 1024
    conditional_video_equi = torch.nn.functional.interpolate(conditional_video_equi, size=(512, 1024),
                                                              mode='bilinear', align_corners=False)
    mask = torch.nn.functional.interpolate(mask.float(), size=(512, 1024), mode='nearest').bool()

    return conditional_video_equi, mask

@torch.no_grad()
def get_latents(condition_video_persp, conditional_video_equi, mask, accelerator, settings, latents_path,
                blend_frames=10, num_frames_batch=25, latents_model=None, offload_to_cpu=True, force_get_latents=False):
    if not latents_path.exists() or force_get_latents:
        conditional_video_equi = conditional_video_equi.to(settings.weight_dtype)

        if latents_model is None:
            latents_model = get_360_latents_model(settings.paths.unet_path, accelerator, len(conditional_video_equi),
                                                  accelerator.device, settings.weight_dtype, blend_frames=blend_frames,
                                                  num_frames_batch=num_frames_batch)
        if len(condition_video_persp) > latents_model.argus_config.num_frames:
            print("WARNING: video longer than max frames for argus latents model, updating max frames")
            latents_model.argus_config.num_frames = len(condition_video_persp)
        
        if offload_to_cpu:
            condition_video_persp = condition_video_persp.cpu()
            conditional_video_equi = conditional_video_equi.cpu()
            mask = mask.cpu()
            
        argus_latents, _ = latents_model.forward_argus(condition_video_persp, conditional_video_equi, mask,
                                                       return_unet_latents=False)
        argus_latents = argus_latents[0]
        latents_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save({"latents": argus_latents.cpu()}, latents_path)
    else:
        latents_data = torch.load(latents_path)
        argus_latents = latents_data["latents"].to(accelerator.device)

    return argus_latents

def get_full_vid_eq_seg_masks(first_mask_eq, pred_eq_frames_torch, sam_runner, threshold=0.08):
    vid_seg_logits, vid_seg_confs = sam_runner.run_through_video(pred_eq_frames_torch,
                                                                 masks=first_mask_eq[None][None])
    vid_seg_mean_above_conf = vid_seg_confs > threshold
    return vid_seg_mean_above_conf

def get_initial_frame_mask(data, sam_runner, mapper, every_x_points=10):
    """SAM2 mask of the tracked object in frame 0, prompted with every n-th query point."""
    pos_points = mapper.point.vc.to_ij(data.trajectory)[0, 0].cpu().numpy().astype(int).tolist()
    pos_points = pos_points[::every_x_points]
    first_mask = sam_runner.run_through_image(
        (data.video[0, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
        bbox_xyxy=None, positive_points=pos_points, negative_points=None)
    return first_mask, pos_points

def main(args, settings):
    if args.split_file is not None:
        from runners.settings import read_split_file
        settings.specific_video_names = read_split_file(args.split_file)


    _, dl = get_dataset(settings.paths.tapvid360_data_root, settings.ds_name, settings.specific_video_names)
    accelerator = Accelerator(mixed_precision='no')
    device = accelerator.device

    for data in tqdm(dl):
        assert len(data.seq_name) == 1, "Batch size greater than 1 not supported"
        seq_name = data.seq_name[0]
        save_seq_name = seq_name.replace("/", "-")
        
        argus_feat_dir = settings.paths.out_root / "argus_feats" / settings.ds_name / seq_name / f"gt_poses-{args.use_gt_rot}"
        latents_path = argus_feat_dir / "latents.pth"
        rotations_path = argus_feat_dir / "rotations.pth"
        
        results_dir = settings.paths.out_root / "results" / settings.ds_name / f"gt_poses-{args.use_gt_rot}"
        metrics_dir = settings.paths.out_root / "metrics" / settings.ds_name / f"gt_poses-{args.use_gt_rot}"
            
        metrics_file_name = metrics_dir / f"{save_seq_name}_metrics.json"
        
        if metrics_file_name.exists() and settings.skip_if_exists:
            print(f"Skipping {seq_name} as metrics already exist")
            continue

        print(f"\nProcessing: {seq_name}")
        
        # --- Stage 1: Pose & Latents ---
        if not latents_path.exists() or not rotations_path.exists():
            print("  Stage 1: Extracting poses and latents...")
            data.video = data.video.to(device, non_blocking=True)
            normed_vid = (data.video.float() * 2) - 1
            
            data.rotations = data.rotations.to(device, non_blocking=True)
            R_w2c, fov_x = get_rotations(data, device, use_gt=args.use_gt_rot, offload_from_gpu=True)

            # Prepare Argus inputs
            mapper_argus = Mappers(data.video.shape[-1], data.video.shape[-2], settings.argus_eq_height, fov_x=fov_x)
            input_eq_frames, input_masks = create_argus_inputs(data.video[0].permute(0, 2, 3, 1), R_w2c, mapper_argus)
            
            # Extract latents
            latents_model = get_360_latents_model(settings.paths.unet_path, accelerator, len(input_eq_frames),
                                                  device, settings.weight_dtype, blend_frames=10, num_frames_batch=25)
            
            argus_latents = get_latents(normed_vid[0], input_eq_frames, input_masks, accelerator, settings, latents_path,
                                        latents_model=latents_model, force_get_latents=True)
            
            torch.save({"rotations": R_w2c.cpu(), "fov_x": fov_x}, rotations_path)
            
            # Clean up Stage 1
            del latents_model, input_eq_frames, input_masks, normed_vid
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print("  Stage 1: Latents and rotations already exist, loading...")
            latents_data = torch.load(latents_path)
            argus_latents = latents_data["latents"].to(device)
            rotations_data = torch.load(rotations_path)
            R_w2c = rotations_data["rotations"].to(device)
            fov_x = rotations_data["fov_x"]

        # --- Stage 2: Tracking & Metrics ---
        print("  Stage 2: Running tracking and metrics...")
        _t_start = time.perf_counter()
        
        data.video = data.video.to(device, non_blocking=True)
        mapper = Mappers(data.video.shape[-1], data.video.shape[-2], settings.gen_eq_height, fov_x=fov_x)
        
        vae, sam_pred_video, sam_pred_image, cotracker = get_models(settings.paths.sam_checkpoint, accelerator,
                                                                    device, debug_skip_vae=False)
        sam_runner = SAMRunner(sam_img_pred=sam_pred_image, sam_video_pred=sam_pred_video)
        
        argus_latents = argus_latents.to(device)  # forward_argus offloads to CPU when offload_to_cpu=True
        with torch.no_grad():
            orig_pred_eq_frames_torch = _decode_pred_eq_frames(argus_latents, settings, vae)
        
        orig_pred_eq_frames_torch = _resize_eq_frames(orig_pred_eq_frames_torch,
                                                      new_shape=(settings.gen_eq_height, settings.gen_eq_height * 2))
        
        pred_eq_frames_torch = overlay_orig_persp_on_pred_eq(data, mapper, orig_pred_eq_frames_torch, R_w2c,
                                                             with_erode=True).to(torch.uint8)

        debug_dir = None
        if args.debugs:
            debug_dir = (settings.paths.out_root / "debugs" / settings.ds_name /
                         save_seq_name / f"gt_poses-{args.use_gt_rot}")
            debug_dir.mkdir(parents=True, exist_ok=True)
            _save_eq_frames(orig_pred_eq_frames_torch, debug_dir / "pred_eq_frames")
            _save_eq_frames(pred_eq_frames_torch, debug_dir / "pred_eq_frames_overlaid")

        # Get initial mask
        first_mask, pos_points = get_initial_frame_mask(data, sam_runner, mapper, every_x_points=5)

        first_mask_eq_torch_, _ = mapper.image.perspective_image_to_equirectangular(
            torch.from_numpy(first_mask[None, ..., None]).to(device), R_w2c[0], interp_mode="nearest")
        first_mask_eq = first_mask_eq_torch_[0, ..., 0].to(bool)

        # Propagate mask
        vid_seg_mean_above_conf = get_full_vid_eq_seg_masks(first_mask_eq, pred_eq_frames_torch, sam_runner,
                                                            threshold=0.08)

        if debug_dir is not None:
            _save_sam_overlays(pred_eq_frames_torch, vid_seg_mean_above_conf[0],
                                debug_dir / "sam_masks")

        # Object-centered perspective tracking
        obj_cnt_persp_imgs, obj_cnt_rot_matrices = get_object_centred_persp_imgs_with_interpolation(
            pred_eq_frames_torch, vid_seg_mean_above_conf, mapper, batchsize=1)

        if debug_dir is not None:
            _save_obj_centre_frames(obj_cnt_persp_imgs[0], debug_dir / "obj_centre_persp")

        cotracker_input_frames = torch.cat([data.video[:, 0:1].permute(0, 1, 3, 4, 2) * 255, obj_cnt_persp_imgs[:, 1:]],
                                           dim=1)
        
        query_frame_points = mapper.point.vc.to_ij(data.trajectory)[0, 0]
        
        pred_tracks, _ = run_through_cotracker(cotracker, 1, device,
                                                      cotracker_input_frames, query_frame_points.flip(-1))
        
        ct_tracks, _ = run_through_cotracker(cotracker, 1, device,
                                                   data.video.permute(0, 1, 3, 4, 2) * 255, query_frame_points.flip(-1))
        
        # Mapping back
        eq_points_pred = mapper.point.ij.to_cr(pred_tracks, obj_cnt_rot_matrices)
        pred_unit_vectors = mapper.point.cr.to_vc(eq_points_pred[0].to(device), R_w2c)
        pred_unit_vectors[0] = data.trajectory[0, 0]
        pred_points = mapper.point.vc.to_ij(pred_unit_vectors)
        
        pred_in_front_of_camera = ~(pred_unit_vectors[..., 0] < 0.1).any(1)
        points_in_frame = (((pred_points[..., 0] < data.video.shape[-1]) & (
                pred_points[..., 1] < data.video.shape[-2])).all(-1)) & pred_in_front_of_camera

        final_results = pred_unit_vectors.clone()
        ct_preds = mapper.point.ij.to_vc(ct_tracks)
        final_results[points_in_frame] = ct_preds[0, points_in_frame]

        # Metrics
        metrics = compute_metrics(final_results, data.trajectory[0], data.visibility[0])
        results_dir.mkdir(exist_ok=True, parents=True)
        metrics_dir.mkdir(exist_ok=True, parents=True)
        torch.save(final_results.cpu(), results_dir / f"{save_seq_name}.pth")
        with open(metrics_file_name, 'w') as f:
            json.dump({m: metrics[m].mean().item() for m in metrics}, f, indent=4)
            
        motion_metrics = compute_metrics_by_motion_type(
            final_results.cpu(), data.trajectory[0].cpu(),
            data.visibility[0].cpu(), R_w2c.cpu())
        with open(metrics_dir / f"{save_seq_name}_motion_metrics.json", 'w') as f:
            json.dump(motion_metrics, f, indent=4)
            
        print(f"  Inference time: {time.perf_counter() - _t_start:.2f}s")
        for m in metrics:
            print(f"    {m}: {metrics[m].mean().item():.4f}")

        # Clean up
        del vae, sam_pred_video, sam_pred_image, cotracker, argus_latents, R_w2c, data
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    args = get_args()
    settings = load_config(args.config)
    main(args, settings)

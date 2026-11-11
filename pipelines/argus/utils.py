import torch
from diffusers import UNetSpatioTemporalConditionModel, AutoencoderKLTemporalDecoder
from transformers import CLIPVisionModelWithProjection

from pipelines.argus.custom_argus_pipeline import StableVideoDiffusionPipelineCustom


def get_models_custom(args, accelerator, weight_dtype):
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        pretrained_model_name_or_path=args.unet_path,
        subfolder="unet",
    )

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)

    pipeline = StableVideoDiffusionPipelineCustom.from_pretrained(
        args.pretrained_model_name_or_path,
        # torch_dtype below does not convert components that are already loaded, so cast them here.
        unet=accelerator.unwrap_model(unet).to(weight_dtype),
        image_encoder=accelerator.unwrap_model(image_encoder).to(weight_dtype),
        vae=accelerator.unwrap_model(vae).to(weight_dtype),
        revision=args.revision,
        torch_dtype=weight_dtype,
    ).to(accelerator.device)

    # Low-memory mode: trades some speed for a lower UNet memory peak. float32 keeps the paper's setup.
    if weight_dtype != torch.float32:
        pipeline.unet.enable_forward_chunking()

    return pipeline
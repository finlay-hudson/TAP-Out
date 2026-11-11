"""
Altered to allow for frames to just be inputted and not needing a directory
"""

from collections import OrderedDict
import types
from pathlib import Path
from typing import Tuple

import torch

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


@torch.inference_mode()
def init_state(
        self,
        images,
        video_height,
        video_width,
        compute_device,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
):
    """Initialize an inference state."""
    inference_state = {}
    inference_state["images"] = images
    inference_state["num_frames"] = len(images)
    # whether to offload the video frames to CPU memory
    # turning on this option saves the GPU memory with only a very small overhead
    inference_state["offload_video_to_cpu"] = offload_video_to_cpu
    # whether to offload the inference state to CPU memory
    # turning on this option saves the GPU memory at the cost of a lower tracking fps
    # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
    # and from 24 to 21 when tracking two objects)
    inference_state["offload_state_to_cpu"] = offload_state_to_cpu
    # the original video height and width, used for resizing final output scores
    inference_state["video_height"] = video_height
    inference_state["video_width"] = video_width
    inference_state["device"] = compute_device
    if offload_state_to_cpu:
        inference_state["storage_device"] = torch.device("cpu")
    else:
        inference_state["storage_device"] = compute_device
    # inputs on each frame
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    # visual features on a small number of recently visited frames for quick interactions
    inference_state["cached_features"] = {}
    # values that don't change across frames (so we only need to hold one copy of them)
    inference_state["constants"] = {}
    # mapping between client-side object id and model-side object index
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    # A storage to hold the model's tracking results and states on each frame
    inference_state["output_dict"] = {
        "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
    }
    # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
    inference_state["output_dict_per_obj"] = {}
    # A temporary storage to hold new outputs when user interact with a frame
    # to add clicks or mask (it's merged into "output_dict" before propagation starts)
    inference_state["temp_output_dict_per_obj"] = {}
    # Frames that already holds consolidated outputs from click or mask inputs
    # (we directly use their consolidated outputs during tracking)
    inference_state["consolidated_frame_inds"] = {
        "cond_frame_outputs": set(),  # set containing frame indices
        "non_cond_frame_outputs": set(),  # set containing frame indices
    }
    # metadata for each tracking frame (e.g. which direction it's tracked)
    inference_state["tracking_has_started"] = False
    inference_state["frames_already_tracked"] = {}
    inference_state["frames_tracked_per_obj"] = {}

    # Warm up the visual backbone and cache the image feature on frame 0
    self._get_image_feature(inference_state, frame_idx=0, batch_size=1)

    return inference_state


def get_sam2_predictor(single_image: bool = False, model_cfg: str = "sam2_hiera_l.yaml",
                       checkpoint: str | Path = "sam2_hiera_large.pt", device: torch.device = torch.device("cuda")):
    if single_image:
        return SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))
    pred = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    # Monkey patch the new init state method to allow for frames to just be fed to the method
    pred.init_state = types.MethodType(init_state, pred)

    return pred


def sam_preprocessing(frames: torch.Tensor, mask: torch.Tensor, resize_to: tuple) -> Tuple[
    torch.Tensor, torch.Tensor]:
    if not len(frames):
        raise ValueError("No frames given")
    img_mean = (0.485, 0.456, 0.406)
    img_std = (0.229, 0.224, 0.225)

    frames_tens = frames.permute(0, 3, 1, 2)

    frames_tens_scaled = frames_tens / 255.0
    frames_tens_scaled = torch.nn.functional.interpolate(frames_tens_scaled, size=resize_to, mode='bilinear',
                                                         align_corners=False)

    # normalize by mean
    frames_tens_scaled -= torch.tensor(img_mean, dtype=torch.float32)[:, None, None].to(frames_tens_scaled.device)

    # normalize by std
    frames_tens_scaled /= torch.tensor(img_std, dtype=torch.float32)[:, None, None].to(frames_tens_scaled.device)

    return frames_tens_scaled, torch.nn.functional.interpolate(mask.float(), size=resize_to, mode='bilinear',
                                                               align_corners=False)[:, 0]

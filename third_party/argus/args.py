"""Argus (SVD) sampling arguments.

Adapted from the Argus inference script (https://github.com/Red-Fairy/argus-code).
Not covered by TAP-Out's MIT license — see THIRD_PARTY_NOTICES.md.
TAP-Out builds these via ``runners.model_utils.get_360_latents_model``.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SVDArguments:
    """
    A class to hold all configuration arguments for the SVD sampling script.
    """

    # --- Model and Path Arguments ---
    # These arguments specify the paths to models, data, and output folders.
    unet_path: str  # This is a required path to the pretrained U-Net model.
    pretrained_model_name_or_path: str = 'stabilityai/stable-video-diffusion-img2vid'
    revision: Optional[str] = None
    val_base_folder: List[str] = field(default_factory=list)
    val_clip_file: Optional[str] = None
    calibration_cache_path: Optional[str] = None
    cached_motion_npz_path: Optional[str] = None
    val_save_folder: str = 'results'
    cached_motion_path: Optional[str] = None

    # --- Data Arguments ---
    # These arguments control the input data's properties, such as size and format.
    dataset_size: Optional[int] = None
    width: int = 1024
    height: int = 512
    equirectangular_input: bool = False
    crop_center_then_calibrate: bool = False
    narrow: bool = False
    width_narrow: int = 512
    height_narrow: int = 512

    # --- Camera and FOV Arguments ---
    # These arguments relate to the virtual camera's field of view (FOV) and orientation.
    fov_x_min: float = 90.0
    fov_x_max: float = 90.0
    fov_y_min: float = 90.0
    fov_y_max: float = 90.0
    fixed_fov: Optional[float] = None
    fixed_rpy: bool = False
    yaw_start: float = 0.0
    noisy_rpy: float = 0.0

    # --- Sampling and Frame Arguments ---
    # These arguments define the video's temporal properties, like frame count and rate.
    frame_rate: Optional[int] = None
    frame_interval: int = 6
    fixed_start_frame: bool = False
    full_sampling: bool = False
    num_frames: int = 25
    num_frames_batch: int = 25
    blend_frames: int = 0

    # --- Inference Arguments ---
    # Core parameters for the diffusion model's generation process.
    num_inference_steps: int = 50
    motion_bucket_id: int = 127
    noise_aug_strength: float = 0.02
    guidance_scale: float = 1.0
    decode_chunk_size: int = 10
    noise_conditioning: bool = False

    # --- Advanced Generation Arguments ---
    # Special techniques used during or after inference.
    rotation_during_inference: bool = False
    post_rotation: bool = False
    inference_final_rotation: int = 0
    blend_decoding_ratio: int = 4
    extended_decoding: bool = False
    replacement_sampling: bool = False
    calibration_cache_path_add_prefix: bool = False

    # --- Calibration Arguments ---
    # Settings for the camera motion prediction and calibration process.
    predict_camera_motion: bool = False
    dense_calibration: bool = False
    calibration_img_size: int = 512

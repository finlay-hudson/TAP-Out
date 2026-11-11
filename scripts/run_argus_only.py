"""Smoke-test Argus 360° out-painting on a video.

Runs just Stage 1 of the TAP-Out pipeline:
    VGGT camera poses → perspective→equirectangular → Argus U-Net → VAE decode

Skips SAM2 and CoTracker entirely. Handy for eyeballing whether Argus
generalises to your video before wiring up the full tracker.

Outputs into ``--out``:
    * ``equirectangular_pred.mp4`` — the generated 360° video (equirectangular).
    * ``equirectangular_input.mp4`` — the masked input Argus saw (for reference).
    * ``side_by_side.mp4`` — input | output stacked, easy to sanity-check.
    * ``latents.pth`` — cached Argus latents (delete to force recompute).

Example
-------
    python scripts/run_argus_only.py \\
        --video /path/to/clip.mp4 \\
        --fov_x 40 \\
        --num_frames 25 \\
        --out outputs/argus_only/clip
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from accelerate import Accelerator
from diffusers import AutoencoderKLTemporalDecoder
from natsort import natsorted
from PIL import Image
from torchvision import transforms

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conversions.mapper import Mappers  # noqa: E402
from runners.settings import load_config  # noqa: E402
from runners.model_utils import decode_latents, get_360_latents_model  # noqa: E402
from runners.pose_utils import VGGTPoseRunner  # noqa: E402
from run_tapout_release import create_argus_inputs, get_latents  # noqa: E402


def _load_frames(video_path: Path, num_frames: int | None) -> torch.Tensor:
    if video_path.is_dir():
        img_files = natsorted(list(video_path.glob("*.jpg")) + list(video_path.glob("*.png")))
        assert img_files, f"No images found in {video_path}"
        tfm = transforms.ToTensor()
        frames = []
        for f in img_files:
            frames.append(tfm(Image.open(f).convert("RGB")))
            if num_frames and len(frames) >= num_frames:
                break
    else:
        cap = cv2.VideoCapture(str(video_path))
        tfm = transforms.ToTensor()
        frames = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(tfm(Image.fromarray(rgb)))
            if num_frames and len(frames) >= num_frames:
                break
        cap.release()
    return torch.stack(frames)


def _write_mp4(frames_uint8: np.ndarray, path: Path, fps: int = 4):
    T, H, W = frames_uint8.shape[:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for t in range(T):
        bgr = cv2.cvtColor(frames_uint8[t], cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()


def _stack_side_by_side(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.concatenate([a, b], axis=1)


def run(video_path: Path, fov_x_deg: float, num_frames: int, out_dir: Path,
        settings, fps: int = 4, force_recompute: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(mixed_precision="no")
    device = accelerator.device

    print("Loading frames…")
    video = _load_frames(video_path, num_frames).to(device)
    T, _, H, W = video.shape
    print(f"  {T} frames at {W}×{H}")

    # ------------------------------------------------------------------ VGGT
    print("Estimating camera poses with VGGT…")
    pose_runner = VGGTPoseRunner(device=device)
    pred_extrinsics, _ = pose_runner.run(video.permute(0, 2, 3, 1))
    pred_rots = pose_runner.convert_to_tapvid360_format(
        pred_extrinsics[..., :3]).to(torch.float32)[0]
    R_w2c = pose_runner.align_to_ground_truth_start_rotations(
        pred_rots, torch.eye(3).to(pred_rots[0].device))
    fov_x = torch.tensor(fov_x_deg, device=device)
    del pose_runner
    torch.cuda.empty_cache()

    # ------------------------------------------------------------- Argus in
    print("Building perspective→equirectangular Argus input…")
    mapper_argus = Mappers(W, H, settings.argus_eq_height, fov_x=fov_x)
    input_eq_frames, input_masks = create_argus_inputs(
        video.permute(0, 2, 3, 1), R_w2c, mapper_argus)

    input_eq_uint8 = (((input_eq_frames.clamp(-1, 1) + 1) * 127.5)
                      .to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())
    _write_mp4(input_eq_uint8, out_dir / "equirectangular_input.mp4", fps=fps)

    # ------------------------------------------------------------ Argus run
    print("Running Argus (this is the slow bit)…")
    normed_vid = video.float() * 2 - 1
    latents_model = get_360_latents_model(
        settings.paths.unet_path, accelerator, len(input_eq_frames), device,
        settings.weight_dtype, blend_frames=settings.blend_frames,
        num_frames_batch=settings.num_frames_batch,
    )
    latents_path = out_dir / "latents.pth"
    argus_latents = get_latents(
        normed_vid, input_eq_frames, input_masks, accelerator, settings,
        latents_path, latents_model=latents_model, offload_to_cpu=settings.offload_to_cpu,
        force_get_latents=force_recompute,
    )
    del latents_model, normed_vid
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- VAE
    print("Loading SVD VAE and decoding latents…")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        subfolder="vae", revision=None,
    ).to(device).eval()
    argus_latents = argus_latents.to(device)  # forward_argus offloads back to CPU when offload_to_cpu=True
    with torch.no_grad():
        pred = decode_latents(
            vae, argus_latents[None], argus_latents.shape[0],
            settings.decode_chunk_size, extended_decoding=False,
        )
    # (1, C, T, H, W) → (T, H, W, C) uint8
    pred_uint8 = (((pred[0].permute(1, 2, 3, 0).clamp(-1, 1) + 1) * 127.5)
                  .to(torch.uint8).cpu().numpy())
    _write_mp4(pred_uint8, out_dir / "equirectangular_pred.mp4", fps=fps)

    # ------------------------------------------------------------- Side by side
    # Resize input to match output height so we can stack.
    T_out, H_out, W_out, _ = pred_uint8.shape
    resized_input = np.stack([
        cv2.resize(input_eq_uint8[t], (W_out, H_out), interpolation=cv2.INTER_LINEAR)
        for t in range(T_out)
    ])
    side_by_side = np.stack([
        _stack_side_by_side(resized_input[t], pred_uint8[t]) for t in range(T_out)
    ])
    _write_mp4(side_by_side, out_dir / "side_by_side.mp4", fps=fps)

    print(f"\nSaved: {out_dir}/equirectangular_pred.mp4")
    print(f"       {out_dir}/equirectangular_input.mp4")
    print(f"       {out_dir}/side_by_side.mp4")


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--fov_x", type=float, default=75.18,
                   help="Horizontal FoV of the input camera in degrees.")
    p.add_argument("--num_frames", type=int, default=32,
                   help="Cap on frames (Argus was trained on 32).")
    p.add_argument("--out", type=Path, default=Path("outputs/argus_only"))
    p.add_argument("--config", "-c", type=str, default=None)
    p.add_argument("--fps", type=int, default=4,
                   help="Playback fps for the output MP4s.")
    p.add_argument("--force", action="store_true",
                   help="Recompute latents even if a cache exists.")
    return p.parse_args()


if __name__ == "__main__":
    args = _cli()
    settings = load_config(args.config)
    run(args.video, args.fov_x, args.num_frames, args.out, settings,
        fps=args.fps, force_recompute=args.force)

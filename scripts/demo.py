"""Run TAP-Out on a single perspective video.

Given an MP4 (or folder of frames) and a JSON of query pixel points in the
first frame, produces:

* ``<out>/tracks_pixels.npy`` — (T, N, 2) tracked pixel positions per frame.
* ``<out>/tracks_unit_vectors.npy`` — (T, N, 3) 3D direction vectors per frame
  in the camera coordinate frame at time t (TAPVid-360 convention).
* ``<out>/tracks_visibility.npy`` — (T, N) bool visibility mask.
* ``<out>/preview.mp4`` — a debug overlay if ``--preview`` is set.

Example
-------
    python scripts/demo.py \
        --video path/to/clip.mp4 \
        --queries '[[512, 300], [640, 400]]' \
        --fov_x 75.18 \
        --out outputs/demo/clip

Assumes the query points are given in the first frame's pixel coordinates.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from accelerate import Accelerator
from natsort import natsorted
from PIL import Image
from torchvision import transforms

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conversions.mapper import Mappers  # noqa: E402
from data.data_utils import DataSample  # noqa: E402
from runners.settings import load_config  # noqa: E402
from runners.model_utils import SAMRunner, get_models, get_360_latents_model  # noqa: E402
from runners.pose_utils import VGGTPoseRunner  # noqa: E402
from runners.pipeline_helpers import (  # noqa: E402
    _decode_pred_eq_frames,
    _resize_eq_frames,
    run_through_cotracker,
)
from runners.utils import (  # noqa: E402
    get_object_centred_persp_imgs_with_interpolation,
    overlay_orig_persp_on_pred_eq,
)
from run_tapout_release import (  # noqa: E402
    create_argus_inputs,
    get_full_vid_eq_seg_masks,
    get_latents,
)


def _load_frames(video_path: Path, num_frames: int | None) -> torch.Tensor:
    """Return (T, C, H, W) float tensor in [0, 1]."""
    if video_path.is_dir():
        img_files = natsorted(video_path.glob("*.jpg")) + natsorted(video_path.glob("*.png"))
        assert img_files, f"No images found in {video_path}"
        frames = []
        tfm = transforms.ToTensor()
        for f in img_files:
            frames.append(tfm(Image.open(f).convert("RGB")))
            if num_frames and len(frames) >= num_frames:
                break
    else:
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        tfm = transforms.ToTensor()
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


def _parse_queries(spec: str) -> np.ndarray:
    if spec.endswith(".json"):
        with open(spec) as f:
            data = json.load(f)
    else:
        data = json.loads(spec)
    arr = np.array(data, dtype=np.float32)
    assert arr.ndim == 2 and arr.shape[1] == 2, "queries must be shape (N, 2) [x, y]"
    return arr


def _write_preview(video: torch.Tensor,
                   source_pixels: np.ndarray, in_view: np.ndarray,
                   eq_pixels: np.ndarray, in_front: np.ndarray,
                   equirectangular_frames: np.ndarray, out: Path):
    """Native TAP-Out visualisation: two-panel video.

    Top row  : source perspective frame with in-view dots (green, tab20 palette).
    Bottom   : Argus-outpainted 360° panorama with a dot at the tracked 3D
               direction's equirectangular projection (green when in-view of
               the source, yellow when out-of-view but still ahead of camera).

    Args
    ----
    video : (T, C, H, W) source frames in [0, 1]
    source_pixels : (T, N, 2) — per-frame dot position in source pixels
    in_view : (T, N) bool — source-side dots drawn only where True
    eq_pixels : (T, N, 2) — per-frame dot position on the equirectangular canvas
    in_front : (T, N) bool — panorama dots drawn only where True
    equirectangular_frames : (T, H_eq, W_eq, 3) uint8 RGB (Argus output)
    """
    T, N, _ = source_pixels.shape
    H, W = video.shape[-2:]
    H_eq, W_eq = equirectangular_frames.shape[1:3]

    # tab20 palette hard-coded so we don't drag in matplotlib. BGR order.
    _TAB20_BGR = [
        (180, 119, 31), (232, 199, 174), (14, 127, 255), (120, 187, 255),
        (44, 160, 44),  (138, 223, 152), (40, 39, 214),  (150, 152, 255),
        (189, 103, 148),(213, 176, 197), (75, 86, 140),  (148, 156, 196),
        (194, 119, 227),(210, 182, 247), (127, 127, 127),(199, 199, 199),
        (34, 189, 188), (141, 219, 219), (207, 190, 23), (229, 218, 158),
    ]
    colours = [_TAB20_BGR[i % len(_TAB20_BGR)] for i in range(N)]

    OOV_TINT = 0.55  # dim the query colour a touch when point is OOV of source

    src_scale = W_eq / W
    src_H_scaled = int(round(H * src_scale))
    canvas_W = W_eq
    canvas_H = src_H_scaled + H_eq
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 8, (canvas_W, canvas_H))

    for t in range(T):
        src_bgr = cv2.cvtColor(
            (video[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR)
        src_bgr = cv2.resize(src_bgr, (canvas_W, src_H_scaled), interpolation=cv2.INTER_AREA)
        eq_bgr = cv2.cvtColor(equirectangular_frames[t], cv2.COLOR_RGB2BGR)

        for i in range(N):
            col = colours[i]
            # Source-side dot: only when tracked point is in the perspective view.
            if in_view[t, i]:
                x_src = int(round(source_pixels[t, i, 0] * src_scale))
                y_src = int(round(source_pixels[t, i, 1] * src_scale))
                cv2.circle(src_bgr, (x_src, y_src), 10, col, -1)
                cv2.circle(src_bgr, (x_src, y_src), 10, (0, 0, 0), 2)

            # Equirectangular dot: any frame the point is in front of the camera.
            if in_front[t, i]:
                x_eq = int(round(eq_pixels[t, i, 0])) % W_eq
                y_eq = int(round(np.clip(eq_pixels[t, i, 1], 0, H_eq - 1)))
                eq_col = col if in_view[t, i] else tuple(int(c * OOV_TINT) for c in col)
                # Larger hollow ring for OOV so it stands out on the panorama.
                if in_view[t, i]:
                    cv2.circle(eq_bgr, (x_eq, y_eq), 12, eq_col, -1)
                else:
                    cv2.circle(eq_bgr, (x_eq, y_eq), 14, eq_col, 3)
                cv2.circle(eq_bgr, (x_eq, y_eq), 12, (0, 0, 0), 2)

        canvas = np.vstack([src_bgr, eq_bgr])
        cv2.putText(canvas, f"t={t}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, "Source (in-view: solid dot)", (20, src_H_scaled - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Argus 360 panorama (in-view: solid, OOV: ring)",
                    (20, src_H_scaled + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()


def run_demo(
    video_path: Path,
    queries_xy: np.ndarray,
    fov_x_deg: float,
    out_dir: Path,
    settings,
    num_frames: int = 32,
    preview: bool = False,
    num_inference_steps: int = 50,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(mixed_precision="no")
    device = accelerator.device

    print("Loading frames…")
    video = _load_frames(video_path, num_frames).to(device)  # (T, C, H, W), [0, 1]
    T, _, H, W = video.shape
    print(f"  {T} frames at {W}x{H}")

    # -- Camera poses via VGGT --------------------------------------------------
    print("Estimating camera poses (VGGT)…")
    pose_runner = VGGTPoseRunner(device=device)
    pred_extrinsics, _ = pose_runner.run(video.permute(0, 2, 3, 1))
    pred_rots = pose_runner.convert_to_tapvid360_format(
        pred_extrinsics[..., :3]).to(torch.float32)[0]
    R_w2c = pose_runner.align_to_ground_truth_start_rotations(
        pred_rots, torch.eye(3).to(pred_rots[0].device))
    fov_x = torch.tensor(fov_x_deg, device=device)
    del pose_runner
    torch.cuda.empty_cache()

    # -- Argus latents ----------------------------------------------------------
    print("Running Argus 360° outpainting…")
    mapper_argus = Mappers(W, H, settings.argus_eq_height, fov_x=fov_x)
    input_eq_frames, input_masks = create_argus_inputs(
        video.permute(0, 2, 3, 1), R_w2c, mapper_argus)

    normed_vid = video.float() * 2 - 1
    latents_model = get_360_latents_model(
        settings.paths.unet_path, accelerator, len(input_eq_frames), device,
        settings.weight_dtype, blend_frames=settings.blend_frames,
        num_frames_batch=settings.num_frames_batch,
        num_inference_steps=num_inference_steps,
    )
    latents_path = out_dir / "latents.pth"
    argus_latents = get_latents(
        normed_vid, input_eq_frames, input_masks, accelerator, settings,
        latents_path, latents_model=latents_model, force_get_latents=False,
    )
    del latents_model, input_eq_frames, input_masks, normed_vid
    gc.collect()
    torch.cuda.empty_cache()

    # -- Decode + overlay -------------------------------------------------------
    print("Decoding equirectangular frames…")
    mapper = Mappers(W, H, settings.gen_eq_height, fov_x=fov_x)
    vae, sam_pred_video, sam_pred_image, cotracker = get_models(
        settings.paths.sam_checkpoint, accelerator, device, debug_skip_vae=False)
    sam_runner = SAMRunner(sam_img_pred=sam_pred_image, sam_video_pred=sam_pred_video)

    argus_latents = argus_latents.to(device)  # get_latents leaves them on CPU when offload_to_cpu=True
    with torch.no_grad():
        orig_pred_eq_frames = _decode_pred_eq_frames(argus_latents, settings, vae)
    orig_pred_eq_frames = _resize_eq_frames(
        orig_pred_eq_frames,
        new_shape=(settings.gen_eq_height, settings.gen_eq_height * 2),
    )

    # DataSample-lite so overlay_orig_persp_on_pred_eq works.
    data = DataSample(video=video.unsqueeze(0), trajectory=torch.zeros(1, T, 1, 3))
    pred_eq_frames = overlay_orig_persp_on_pred_eq(
        data, mapper, orig_pred_eq_frames, R_w2c, with_erode=True).to(torch.uint8)

    # -- SAM2 initial mask from user query points -------------------------------
    print("Prompting SAM2 with query points…")
    first_frame_np = (video[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    first_mask = sam_runner.run_through_image(
        first_frame_np, positive_points=queries_xy.tolist())

    first_mask_eq_torch, _ = mapper.image.perspective_image_to_equirectangular(
        torch.from_numpy(first_mask[None, ..., None]).to(device),
        R_w2c[0], interp_mode="nearest")
    first_mask_eq = first_mask_eq_torch[0, ..., 0].to(bool)

    print("Propagating mask with SAM2 on equirectangular…")
    vid_seg = get_full_vid_eq_seg_masks(
        first_mask_eq, pred_eq_frames, sam_runner, threshold=0.08)

    # -- Object-centred CoTracker3 ---------------------------------------------
    print("Running CoTracker3 on object-centred crops…")
    obj_cnt_persp_imgs, obj_cnt_rot_matrices = \
        get_object_centred_persp_imgs_with_interpolation(
            pred_eq_frames, vid_seg, mapper, batchsize=1)

    ct_input = torch.cat([
        video[:1].permute(0, 2, 3, 1).unsqueeze(0) * 255,
        obj_cnt_persp_imgs[:, 1:],
    ], dim=1)

    # Bypass the CoTracker wrapper's resize+coord-rescale path, which is buggy
    # for non-square input (assumes W==H and does a spurious axis-flip).
    # Passing interp_shape=None makes CoTracker3 handle the native resolution.
    cotracker.interp_shape = None
    query_pts_xy = torch.from_numpy(queries_xy.astype(np.float32).copy()).to(device)
    pred_tracks, _ = run_through_cotracker(cotracker, 1, device, ct_input, query_pts_xy)
    ct_tracks, ct_vis = run_through_cotracker(
        cotracker, 1, device, video.permute(0, 2, 3, 1).unsqueeze(0) * 255, query_pts_xy)

    # -- Map object-centred perspective tracks back to source view --------------
    print("Mapping tracks back to source pixels…")
    eq_points_pred = mapper.point.ij.to_cr(pred_tracks, obj_cnt_rot_matrices)
    pred_unit_vectors = mapper.point.cr.to_vc(eq_points_pred[0].to(device), R_w2c)

    # In-view frames come straight from raw CoTracker on the source video.
    # OOV frames fall back to the OOV projection through the outpainted panorama.
    ct_preds_vc = mapper.point.ij.to_vc(ct_tracks)                          # (1, T, N, 3)
    pred_pixels_in_frame = ((ct_tracks[0, ..., 0] >= 0) & (ct_tracks[0, ..., 0] < W) &
                            (ct_tracks[0, ..., 1] >= 0) & (ct_tracks[0, ..., 1] < H))  # (T, N)
    ct_visible = ct_vis[0] > 0.5 if ct_vis is not None else pred_pixels_in_frame  # (T, N)
    in_view = ct_visible & pred_pixels_in_frame

    final_unit_vectors = pred_unit_vectors.clone()
    final_unit_vectors[in_view] = ct_preds_vc[0][in_view]
    final_pixels = mapper.point.vc.to_ij(final_unit_vectors)

    # In-view visibility (dot on the source frame). Requires point to project
    # inside the image bounds AND be in front of the camera.
    vis_mask = ((final_pixels[..., 0] >= 0) & (final_pixels[..., 0] < W) &
                (final_pixels[..., 1] >= 0) & (final_pixels[..., 1] < H) &
                (final_unit_vectors[..., 0] > 0)).cpu().numpy()

    # Broader "point is in front of camera" — this is what OOV tracking needs.
    # When True but vis_mask is False the point is out of frame; the preview
    # draws it as a ring on the Argus panorama.
    in_front_mask = (final_unit_vectors[..., 0] > 0).cpu().numpy()

    tracks_pix = final_pixels.cpu().numpy()
    tracks_vec = final_unit_vectors.cpu().numpy()

    np.save(out_dir / "tracks_pixels.npy", tracks_pix)
    np.save(out_dir / "tracks_unit_vectors.npy", tracks_vec)
    np.save(out_dir / "tracks_visibility.npy", vis_mask)
    np.save(out_dir / "tracks_in_front.npy", in_front_mask)
    print(f"\nSaved tracks under {out_dir}")
    print(f"  In-view frames  : {vis_mask.sum()}/{vis_mask.size}")
    print(f"  OOV (in-front)  : {(in_front_mask & ~vis_mask).sum()}/{vis_mask.size}")
    print(f"  Behind camera   : {(~in_front_mask).sum()}/{vis_mask.size}")

    if preview:
        # Equirectangular projection of the tracked unit vectors on the Argus
        # panorama. Unit vectors live in the camera frame at time t; the
        # panorama is aligned with frame 0 (the world frame), so we must
        # transform via R_w2c[t] to get the world-frame equirectangular pixel.
        eq_pixels = torch.stack([
            mapper.point.vc.to_cr(final_unit_vectors[t], R_w2c[t])
            for t in range(final_unit_vectors.shape[0])
        ], dim=0).cpu().numpy()
        eq_frames_np = pred_eq_frames.cpu().numpy()  # (T, H_eq, W_eq, 3), uint8 RGB
        preview_path = out_dir / "preview.mp4"
        _write_preview(video, tracks_pix, vis_mask, eq_pixels, in_front_mask,
                       eq_frames_np, preview_path)
        print(f"Wrote preview: {preview_path}")


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--video", type=Path, required=True,
                   help="Path to an MP4 file or a directory of image frames.")
    p.add_argument("--queries", type=str, required=True,
                   help="JSON list of first-frame [x, y] query points, or a path to a .json file.")
    p.add_argument("--fov_x", type=float, default=75.18,
                   help="Horizontal FoV of the input camera in degrees (default: TAPVid360's).")
    p.add_argument("--out", type=Path, default=Path("outputs/demo"),
                   help="Output directory.")
    p.add_argument("--config", "-c", type=str, default=None,
                   help="YAML config (defaults to configs/default.yaml).")
    p.add_argument("--num_frames", type=int, default=32,
                   help="Cap number of input frames (Argus was trained on 32).")
    p.add_argument("--preview", action="store_true",
                   help="Write a preview.mp4 overlay of the tracks.")
    p.add_argument("--steps", type=int, default=50,
                   help="Argus diffusion denoising steps (default 50, "
                        "bump to 100 for higher-quality panorama at 2x cost).")
    return p.parse_args()


if __name__ == "__main__":
    args = _cli()
    queries = _parse_queries(args.queries)
    settings = load_config(args.config)
    run_demo(args.video, queries, args.fov_x, args.out, settings,
             num_frames=args.num_frames, preview=args.preview,
             num_inference_steps=args.steps)

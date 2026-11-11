"""Build a 3-row showcase MP4 for one TAP-Out clip.

Rows (top → bottom):

    1. Equirectangular input  — observed FoV only, OOV = black.
    2. Argus-filled panorama  — generative outpainting of the 360° scene.
    3. Argus-filled + tracked points  — predicted (filled green) and GT
       (hollow yellow) directions projected onto the panorama.

Requires that the clip was evaluated with ``--debugs`` first (
so the per-frame debug assets are on disk)::

    python scripts/evaluate.py --split_file one_clip.txt --use_gt_rot --debugs
    python scripts/visualise_showcase.py \\
        --clip kL39iNxt9QY_clip_0144/0 \\
        --use_gt_rot

Writes a single ``showcase.mp4`` under
``<out_root>/viz/showcase/<ds>/<clip>-<query>/gt_poses-<X>/``.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from runners.settings import load_config  # noqa: E402

EQUI_W, EQUI_H = 720, 360
LABEL_H = 24

POINT_PRED = (0, 230, 120)
POINT_GT = (255, 215, 0)


def _compute_fov_mask(R, fov_x_deg, w_persp, h_persp, w_equi, h_equi):
    """Boolean mask of equi pixels inside the perspective FoV at rotation R."""
    fov_y_deg = fov_x_deg * h_persp / w_persp
    hx = math.radians(fov_x_deg) / 2
    hy = math.radians(fov_y_deg) / 2
    us = np.arange(w_equi)
    vs = np.arange(h_equi)
    theta = us / w_equi * 2 * math.pi - math.pi
    phi = math.pi / 2 - vs / h_equi * math.pi
    cp = np.cos(phi)
    vw = np.stack([
        cp[:, None] * np.cos(theta)[None, :],
        cp[:, None] * np.sin(theta)[None, :],
        np.broadcast_to(np.sin(phi)[:, None], (h_equi, w_equi)),
    ], axis=-1)
    R_np = R.numpy() if isinstance(R, torch.Tensor) else np.asarray(R)
    vc = vw @ R_np.T
    return (
        (vc[..., 0] > 1e-6)
        & (np.abs(np.arctan2(vc[..., 1], vc[..., 0])) <= hx)
        & (np.abs(np.arcsin(np.clip(vc[..., 2], -1, 1))) <= hy)
    )


def _project_vc_to_equi(vc, w_equi, h_equi):
    """Project (N, 3) *world-frame* directions into (N, 2) equi pixels."""
    th = np.arctan2(vc[..., 1], vc[..., 0])
    ph = np.arcsin(np.clip(vc[..., 2], -1, 1))
    u = w_equi / (2 * math.pi) * (th + math.pi)
    v = h_equi / math.pi * (math.pi / 2 - ph)
    return np.stack([u, v], axis=-1)


def _label(img, text):
    """Prepend a small label bar above the image."""
    out = Image.new("RGB", (img.width, img.height + LABEL_H), (20, 20, 20))
    out.paste(img, (0, LABEL_H))
    dr = ImageDraw.Draw(out)
    dr.text((8, 4), text, fill=(220, 220, 220))
    return out


def _encode_mp4(frames, out_path, fps):
    staging = out_path.parent / f".{out_path.stem}_frames"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for i, img in enumerate(frames):
        img.save(staging / f"{i:04d}.png")
    subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", str(staging / "%04d.png"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            str(out_path),
        ],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    shutil.rmtree(staging, ignore_errors=True)
    print(f"  wrote {out_path.name} ({out_path.stat().st_size // 1024} KB)")


def main():
    # Rewrite "--clip <name>" to "--clip=<name>" so clip ids starting with "-"
    # (e.g. "-bpmcBYbTcs_clip_0001/0") don't get mistaken for flags.
    argv = sys.argv[1:]
    for i, tok in enumerate(argv[:-1]):
        if tok in ("--clip", "-c") and not argv[i + 1].startswith("--"):
            argv[i:i + 2] = [f"{tok}={argv[i + 1]}"]
            break

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clip", required=True,
                    help="Clip id in the form '<video>_clip_XXXX/<query>'.")
    ap.add_argument("--config", "-c", default=None, help="Path to YAML config.")
    ap.add_argument("--use_gt_rot", action="store_true",
                    help="Match how evaluate.py was run.")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output dir (default: <out_root>/viz/showcase/...).")
    args = ap.parse_args(argv)

    settings = load_config(args.config)
    gt_tag = f"gt_poses-{args.use_gt_rot}"
    clip, query = args.clip.split("/")
    save_seq_name = args.clip.replace("/", "-")

    dataset_root = settings.paths.tapvid360_data_root / "dataset"
    data_pt = dataset_root / clip / query / "data.pt"

    debug_root = settings.paths.out_root / "debugs" / settings.ds_name / save_seq_name / gt_tag
    equi_overlaid_dir = debug_root / "pred_eq_frames_overlaid"
    equi_outpaint_dir = debug_root / "pred_eq_frames"
    results_pth = (settings.paths.out_root / "results" / settings.ds_name / gt_tag /
                   f"{save_seq_name}.pth")

    missing = [p for p in (data_pt, equi_overlaid_dir, equi_outpaint_dir, results_pth) if not p.exists()]
    if missing:
        sys.exit(
            "Missing inputs — run `evaluate.py --debugs` on this clip first.\n  "
            + "\n  ".join(str(m) for m in missing)
        )

    out_dir = args.out or (settings.paths.out_root / "viz" / "showcase" /
                           settings.ds_name / save_seq_name / gt_tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"clip={args.clip!r}  out={out_dir}  fps={args.fps}")

    data = torch.load(data_pt, weights_only=False)
    R_mats = data["rot_matrices"]
    fov_x = float(data["fov_x"])
    coords = data["coords"].numpy()
    vis = data["coords_visibility"].numpy()
    pred_vc = torch.load(results_pth, weights_only=False)
    pred_vc = pred_vc.numpy() if isinstance(pred_vc, torch.Tensor) else pred_vc

    frames = sorted(int(f.stem) for f in equi_outpaint_dir.glob("*.jpg"))
    query_sel = vis[0]
    stacked = []

    for n in frames:
        # Row 2: full Argus outpaint (shared source for row 1 and row 3 geometry).
        outpaint = Image.open(equi_outpaint_dir / f"{n}.jpg").convert("RGB").resize(
            (EQUI_W, EQUI_H), Image.LANCZOS)
        outpaint_arr = np.array(outpaint)

        # Row 1: observed-only input (same panorama, masked to FoV).
        # We mask against the perspective FoV at the Argus-input resolution to
        # match what the model actually saw.
        mask = _compute_fov_mask(R_mats[n], fov_x, 320, 240, EQUI_W, EQUI_H)
        input_arr = (outpaint_arr * mask[:, :, None]).astype(np.uint8)
        row1 = _label(Image.fromarray(input_arr), "Equirectangular input (observed = colour, OOV = black)")

        # Row 2: Argus-filled.
        row2 = _label(outpaint, "Argus-outpainted panorama")

        # Row 3: outpaint + predicted (filled) + GT (hollow) tracked points.
        row3_img = outpaint.copy()
        dr = ImageDraw.Draw(row3_img)
        R_np = R_mats[n].numpy() if isinstance(R_mats[n], torch.Tensor) else np.asarray(R_mats[n])
        for pc_v, col, radius, filled in (
            (coords[n][query_sel], POINT_GT, 4, False),
            (pred_vc[n][query_sel], POINT_PRED, 3, True),
        ):
            if not len(pc_v):
                continue
            pw = pc_v @ R_np
            ij = _project_vc_to_equi(pw, EQUI_W, EQUI_H)
            for u, v_ in ij:
                box = [u - radius, v_ - radius, u + radius, v_ + radius]
                if filled:
                    dr.ellipse(box, fill=col)
                else:
                    dr.ellipse(box, outline=col, width=2)
        row3 = _label(row3_img, "Argus-outpainted + tracked points (green = pred, yellow = GT)")

        # Stack rows into one tall frame.
        total_h = row1.height + row2.height + row3.height
        canvas = Image.new("RGB", (EQUI_W, total_h), (20, 20, 20))
        canvas.paste(row1, (0, 0))
        canvas.paste(row2, (0, row1.height))
        canvas.paste(row3, (0, row1.height + row2.height))
        stacked.append(canvas)

    _encode_mp4(stacked, out_dir / "showcase.mp4", args.fps)


if __name__ == "__main__":
    main()

# TAP-Out: Tracking Any Point in 360° via Outpainting

Official implementation of **TAP-Out** (BMVC 2026) — a simple, modular pipeline for tracking points that leave the camera's field of view, by first *outpainting* the missing 360° scene with a generative video model and then running a standard 2D tracker on object-centred perspective crops of that panorama.

**Finlay G. C. Hudson, James A. D. Gardner, William A. P. Smith**

[[Project page](https://finlay-hudson.github.io/tapout/)] [[Paper (arXiv:2611.00406)](https://arxiv.org/abs/2611.00406)] [[TAPVid360-10k dataset](https://huggingface.co/datasets/fhudson96/TAPVid360-10k)]

- **Benchmark**: State-of-the-art on TAPVid360-10k, in-view and out-of-view splits.
- **Modularity**: This repo ships with CoTracker3. Tracking is a separate stage, so you can swap in another 2D tracker, but you'll need to add that integration yourself.

```
        ┌──────────────┐   ┌────────────┐   ┌─────────┐   ┌────────────┐   ┌─────────────┐
input → │ VGGT poses   │ → │ Argus 360° │ → │  SAM2   │ → │ Obj-centred│ → │ CoTracker3  │ → tracks
video   │ (extrinsics) │   │ outpaint   │   │ masks   │   │ persp crops│   │ (in-frame)  │
        └──────────────┘   └────────────┘   └─────────┘   └────────────┘   └─────────────┘
```

## Hardware requirements

| GPU VRAM | What works | Notes |
| --- | --- | --- |
| **≥ 24 GB** (RTX 3090/4090, A5000, L4, A40, A100, …) | Paper defaults — `weight_dtype: float32`, 32-frame Argus, 512×1024 equirectangular | Recommended baseline. |
| **16 GB** (tested on RTX 4080) | `--config configs/low_vram.yaml`, up to 32 frames | Uses `float16`, so it isn't the paper's setting. Memory is tight: close other GPU apps first. |
| **< 16 GB** | Not supported | |

Argus is the memory-heavy component (SVD-based diffusion U-Net and VAE). VGGT, SAM2 and CoTracker3 are comparatively cheap.

Set this before running. The 16 GB setup was tested with it:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

If you still run out of memory on 16 GB, lower `pipeline.decode_chunk_size`. With `demo.py` and `run_argus_only.py` you can also lower `--num_frames`; `evaluate.py` always uses the paper's 32 frames.

**Disk and time.** A full install needs about 26 GB: ~8 GB for the Python environment, ~4 GB of checkpoints, and ~14 GB of package and model caches. On an A40 machine, a fresh install takes ~13 minutes and one 32-frame clip takes ~9–13 minutes end to end. To keep caches off a small home directory, set `PIP_CACHE_DIR`/`UV_CACHE_DIR`, `HF_HOME` and `TORCH_HOME`.

## Installation

**One-liner (recommended):**

```bash
git clone https://github.com/finlay-hudson/TAP-Out.git && cd TAP-Out
bash scripts/setup_env.sh                         # uses uv if available, venv+pip otherwise
source .venv/bin/activate
```

Override the CUDA build by setting `TORCH_BACKEND=cu121` / `cu124` / `cpu` before running the script, and the venv location with `VENV_DIR=...`.

<details>
<summary>Manual install</summary>

```bash
python -m venv .venv && source .venv/bin/activate        # or: uv venv --python 3.11
pip install --upgrade pip
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
# VGGT pins numpy<2 upstream, which conflicts with the rest of the stack; it runs fine on numpy 2.
pip install --no-deps "vggt @ git+https://github.com/facebookresearch/vggt.git@44b3afb"
```

`pyproject.toml` pins the SAM2 and CoTracker3 commits used in the paper; VGGT is pinned in the last line above.

</details>

## Checkpoints

One command fetches everything into `./checkpoints/`:

```bash
bash scripts/download_checkpoints.sh
```

This pulls SAM2 (~0.9 GB, direct URL) and the Argus U-Net (~3 GB, Google Drive via `gdown`). VGGT, CoTracker3 and the Stable Video Diffusion VAE/image encoder download automatically on first run via HuggingFace / `torch.hub`. Re-running the script skips files that are already present.

Resulting layout (matches `configs/default.yaml` out of the box):

```
checkpoints/
├── argus/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
└── sam2/
    └── sam2_hiera_large.pt
```

<details>
<summary>Manual fallback if the Google Drive download rate-limits</summary>

Anonymous Google Drive downloads sometimes throttle large files. If `gdown` errors:

1. Open <https://drive.google.com/drive/folders/1mZpViQY2yvwav-CcxdsQouYu8t52b25G> in a browser.
2. Download `config.json` and `diffusion_pytorch_model.safetensors`.
3. Place both in `checkpoints/argus/`.

The Argus checkpoint comes from the [Red-Fairy/argus-code](https://github.com/Red-Fairy/argus-code#-inference) release.

</details>

| Model            | Purpose                        | Source                                                                                                        |
| ---------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| **Argus** (SVD)  | 360° video outpainting         | [Google Drive folder](https://drive.google.com/drive/folders/1mZpViQY2yvwav-CcxdsQouYu8t52b25G) (Argus authors) |
| **SAM2** (Hiera-L) | Object mask + video propagation | [`sam2_hiera_large.pt`](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt)          |
| **VGGT**         | Camera-pose estimation         | HuggingFace (`facebook/VGGT-1B`), auto on first run                                                           |
| **CoTracker3**   | 2D point tracking              | `torch.hub`, auto on first run                                                                                |

Model weights have their own licenses — notably VGGT-1B and CoTracker3 are CC BY-NC 4.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Quick start: end-to-end example

The quickest way to see the whole pipeline working is to run it on one TAPVid360 clip and render the 3-row showcase video shipped at `assets/showcase_kL39iNxt9QY_clip_0144-0.mp4`:

- **Top row:** the observed input projected onto the 360° sphere (everything out of view is black).
- **Middle row:** the Argus outpainted panorama.
- **Bottom row:** the same panorama with TAP-Out's tracked points (green) against ground truth (yellow), including while the object is out of view.

1. Get the TAPVid360-10k data (see [Data](#data) below). Each clip lives at `<tapvid360_data_root>/dataset/<clip>/<query>/` and contains `perspective_frames/` and `data.pt`.
2. Run the pipeline on the clip, keeping the intermediate frames:

   ```bash
   echo 'kL39iNxt9QY_clip_0144/0' > one_clip.txt
   python scripts/evaluate.py --split_file one_clip.txt --use_gt_rot --debugs
   ```

3. Render the showcase:

   ```bash
   python scripts/visualise_showcase.py --clip kL39iNxt9QY_clip_0144/0 --use_gt_rot
   ```

The video is written to `outputs/viz/showcase/tapvid360-10k/kL39iNxt9QY_clip_0144-0/gt_poses-True/showcase.mp4`, and the clip's metrics to `outputs/metrics/tapvid360-10k/gt_poses-True/`. On an A40, step 2 takes about 9 minutes and step 3 a few seconds.

Notes:
- `--use_gt_rot` uses the ground-truth camera poses from `data.pt`. Drop it from both commands to use VGGT-estimated poses instead (the paper's headline setting); the flag must match between the two steps.
- `--debugs` is required: the showcase is built from the intermediate frames it saves.
- Without `--split_file`, `evaluate.py` runs the full 8,887-clip split from the config.
- Use any `<clip>/<query>` from `assets/tapvid360_tapout_subset.txt` to try other clips.
- Pass `--config <your.yaml>` to both commands if your data isn't at `./data_local/TAPVid360-10k`.

## Data

The paper reports numbers on **TAPVid360-TAPOut**, an 8,887-clip subset of TAPVid360-10k that excludes any videos the Argus model trained on (see paper §4).

1. Download the parent dataset from HuggingFace: <https://huggingface.co/datasets/fhudson96/TAPVid360-10k>.
2. Extract it to `./data_local/TAPVid360-10k/` (or update `paths.tapvid360_data_root` in the YAML).
3. The evaluation subset list ships in `assets/tapvid360_tapout_subset.txt` — the loader picks it up automatically.

## Reproducing the paper (evaluation)

```bash
# Full TAPVid360-TAPOut split, VGGT-estimated poses (paper's headline setting).
python scripts/evaluate.py

# With ground-truth poses (upper-bound comparison).
python scripts/evaluate.py --use_gt_rot

# A single clip (JSON array OR one clip per line, e.g. the paper subset at
# assets/tapvid360_tapout_subset.txt is plain text and works directly).
echo 'hDaOt30dHn8_clip_0195/0' > /tmp/one.txt
python scripts/evaluate.py --split_file /tmp/one.txt --use_gt_rot

# Aggregate metrics are written to <out_root>/metrics/<dataset>/.../AGGREGATE.json
```

Per-clip caching (poses + Argus latents) means reruns are cheap — delete `<out_root>/argus_feats/` to force recomputation.

### Reproducibility note

Argus's initial latent uses a pinned RNG seed, so a given clip produces bit-identical panoramas across runs on the same release. Per-clip metrics may differ slightly from the paper table in either direction (the paper run used a different RNG state), but the pipeline is now release-vs-release reproducible.

### Showcase

`assets/showcase_kL39iNxt9QY_clip_0144-0.mp4` is regenerated by the [Quick start](#quick-start-end-to-end-example) commands above. Outpainted regions can differ slightly in texture from the shipped video (see the reproducibility note); the tracks and metrics should match.

### Selecting clips to evaluate

Three ways, in order of precedence:

1. `--split_file <path>` on the CLI (JSON array OR one `clip/query` per line; `#` comments allowed).
2. `eval.specific_video_names` in the YAML config (explicit list).
3. `eval.split_file` in the YAML config (same format as `--split_file`).

With none of those set, the full dataset is evaluated.

### Dumping debug frames

Add `--debugs` to `scripts/evaluate.py` to save per-frame intermediates under
`<out_root>/debugs/<ds>/<clip>-<query>/gt_poses-<X>/`:

- `pred_eq_frames/`           — Argus-outpainted panorama (jpeg per frame).
- `pred_eq_frames_overlaid/`  — same panorama with observed FoV composited back on top (what SAM2 is run against).
- `sam_masks/`                — SAM2 mask overlay on the panorama.
- `obj_centre_persp/`         — the object-centred perspective crop CoTracker3 runs on.

`scripts/visualise_showcase.py` turns these into the 3-row showcase video for any evaluated clip (see [Quick start](#quick-start-end-to-end-example)); its output lands under `<out_root>/viz/showcase/...`.

## Smoke-testing Argus alone

Argus is by far the biggest external component. Before running the full pipeline it's worth checking that its 360° outpainting looks reasonable on your footage:

```bash
python scripts/run_argus_only.py \
    --video path/to/clip.mp4 \
    --fov_x 40 \
    --num_frames 25 \
    --out outputs/argus_only/clip
```

Outputs `equirectangular_pred.mp4` (Argus's 360° prediction), `equirectangular_input.mp4` (what Argus sees), and `side_by_side.mp4`. If the prediction is garbled, tuning `--fov_x` is usually the fix (broadcast footage ~30-50°, phones ~60-75°, action cams up to ~120°).

## Running on your own video (demo)

`scripts/demo.py` runs the full pipeline (VGGT → Argus → SAM2 → CoTracker3) on an arbitrary perspective video and tracks the pixel points you specify on frame 0.

Unlike the Quick start, it needs no ground truth: poses come from VGGT, and you choose the points to track. The output is a 2-row `preview.mp4` (source view + Argus panorama with your points), not the 3-row showcase. As an example, run it on the showcase clip's own perspective frames (no `data.pt` needed — `demo.py` only reads the frames):

```bash
CLIP=path/to/TAPVid360-10k/dataset/kL39iNxt9QY_clip_0144/0/perspective_frames

python scripts/demo.py \
    --video "$CLIP" \
    --queries '[[150, 71], [172, 219], [172, 143], [175, 21], [136, 181], [179, 103]]' \
    --fov_x 75.18 \
    --num_frames 32 \
    --out outputs/demo/kL39iNxt9QY_clip_0144-0 \
    --preview
```

(`--video` accepts an mp4 OR a folder of jpgs/pngs. The six query coordinates above are visible query points from this clip's ground truth at frame 0; for your own video just pass pixel coords you care about.)

Outputs (see `scripts/demo.py`):
- `tracks_pixels.npy` — (T, N, 2) pixel tracks in the source view.
- `tracks_unit_vectors.npy` — (T, N, 3) TAPVid360-style direction vectors.
- `tracks_visibility.npy` — (T, N) in-frame flags.
- `tracks_in_front.npy` — (T, N) flags for points in front of the camera (in view or out of frame).
- `preview.mp4` — two-panel debug overlay (source + Argus panorama with dots), if `--preview`.

Queries can be a JSON literal or a path to a `.json` file with a list of `[x, y]` first-frame points.

## Repository layout

```
TAP-Out/
├── configs/                YAML config (paths + hyperparameters).
├── scripts/
│   ├── setup_env.sh             Bootstrap a uv (or venv) environment.
│   ├── download_checkpoints.sh  Fetch SAM2 + Argus into ./checkpoints/.
│   ├── evaluate.py              Reproduce paper metrics (adds --debugs for intermediates).
│   ├── demo.py                  Run on your own video (MP4 or frame folder).
│   ├── run_argus_only.py        Smoke-test just the 360° outpainting.
│   └── visualise_showcase.py    3-row showcase MP4 (needs --debugs outputs).
├── assets/
│   └── tapvid360_tapout_subset.txt   The 8,887-clip paper split (plain text).
├── conversions/            Perspective ↔ equirectangular ↔ unit-vector maths.
├── data/                   TAPVid360 dataset loader.
├── metrics/                TAPVid360 δ_avg, AD, motion-aware breakdowns.
├── pipelines/              Argus (windowed 360° outpainting) and SAM2 wrappers.
├── runners/                Config loading, model loading, VGGT poses, CoTracker3, helpers.
├── third_party/argus/      Vendored Argus code (see THIRD_PARTY_NOTICES.md).
└── run_tapout_release.py   The end-to-end pipeline function (called by scripts/).
```

## Configuration

All paths and hyperparameters live in `configs/default.yaml`. To use a different config:

```bash
python scripts/evaluate.py --config configs/my_machine.yaml
# or
TAPOUT_CONFIG=configs/my_machine.yaml python scripts/evaluate.py
```

Key fields:
- `paths.*` — model checkpoints and dataset roots.
- `pipeline.gen_eq_height` — equirectangular resolution used for SAM2/CoTracker3 (1024 in the paper).
- `pipeline.weight_dtype` — `float32` (paper) or `float16` (needed on 16 GB GPUs, see `configs/low_vram.yaml`).

Ground-truth vs estimated poses is chosen on the command line, not in the YAML: pass `--use_gt_rot` to `evaluate.py` (and to `visualise_showcase.py`) to reproduce the "GT poses" row of Table 1.

## Citation

```bibtex
@inproceedings{hudson2026tapout,
  title         = {{TAP-Out}: Tracking Any Point in 360$^\circ$ via Outpainting},
  author        = {Hudson, Finlay G. C. and Gardner, James A. D. and Smith, William A. P.},
  booktitle     = {Proceedings of the British Machine Vision Conference (BMVC)},
  year          = {2026},
  eprint        = {2611.00406},
  archivePrefix = {arXiv}
}
```

## Acknowledgements

TAP-Out composes several great open-source projects: [Argus](https://github.com/Red-Fairy/argus-code) (Luo et al., ICCV 2025) for 360° video generation, [SAM2](https://github.com/facebookresearch/sam2) for masks, [VGGT](https://github.com/facebookresearch/vggt) for camera poses, and [CoTracker3](https://github.com/facebookresearch/co-tracker) for in-frame point tracking. TAPVid360-10k is the benchmark from Hudson et al. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for their licenses.

## License

TAP-Out's code is MIT — see [LICENSE](LICENSE). Vendored Argus code and the model weights the pipeline downloads are under their own licenses (some non-commercial); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

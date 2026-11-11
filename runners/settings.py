"""TAP-Out configuration loader.

Loads paths and hyperparameters from YAML — see ``configs/default.yaml`` and
the ``TAPOUT_CONFIG`` environment variable.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path
from typing import Optional

import torch
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "default.yaml"


@dataclasses.dataclass
class Paths:
    unet_path: Path
    sam_checkpoint: Path
    tapvid360_data_root: Path
    out_root: Path


@dataclasses.dataclass
class Settings:
    paths: Paths
    ds_name: str = "tapvid360-10k"
    specific_video_names: Optional[list] = None

    argus_eq_height: int = 512
    gen_eq_height: int = 1024
    weight_dtype: torch.dtype = torch.float32
    decode_chunk_size: int = 10
    offload_to_cpu: bool = True
    skip_if_exists: bool = True
    blend_frames: int = 10
    num_frames_batch: int = 25


_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _resolve(path: str | os.PathLike) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p


def read_split_file(path: str | os.PathLike) -> list[str]:
    """Parse a clip-list file. Accepts either a JSON array (``.json`` or
    content starting with ``[``) or one clip name per line."""
    import json
    text = Path(path).read_text().strip()
    if text.startswith("["):
        return json.loads(text)
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def load_config(config_path: str | os.PathLike | None = None) -> Settings:
    """Load settings from YAML.

    Resolution order: ``config_path`` arg → ``$TAPOUT_CONFIG`` env var →
    ``configs/default.yaml``. Relative paths in the YAML are resolved against
    the repo root, not the caller's CWD, so scripts can be run from anywhere.
    """
    if config_path is None:
        config_path = os.environ.get("TAPOUT_CONFIG", _DEFAULT_CONFIG)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    p = cfg.get("paths", {})
    paths = Paths(
        unet_path=_resolve(p["unet_path"]),
        sam_checkpoint=_resolve(p["sam_checkpoint"]),
        tapvid360_data_root=_resolve(p["tapvid360_data_root"]),
        out_root=_resolve(p["out_root"]),
    )

    pipe = cfg.get("pipeline", {})
    ev = cfg.get("eval", {})
    # eval.specific_video_names wins over eval.split_file if both are set.
    specific_video_names = ev.get("specific_video_names")
    if specific_video_names is None and ev.get("split_file"):
        specific_video_names = read_split_file(_resolve(ev["split_file"]))
    return Settings(
        paths=paths,
        specific_video_names=specific_video_names,
        argus_eq_height=int(pipe.get("argus_eq_height", 512)),
        gen_eq_height=int(pipe.get("gen_eq_height", 1024)),
        weight_dtype=_DTYPE_MAP[str(pipe.get("weight_dtype", "float32")).lower()],
        decode_chunk_size=int(pipe.get("decode_chunk_size", 10)),
        offload_to_cpu=bool(pipe.get("offload_to_cpu", True)),
        skip_if_exists=bool(ev.get("skip_if_exists", True)),
        blend_frames=int(pipe.get("blend_frames", 10)),
        num_frames_batch=int(pipe.get("num_frames_batch", 25)),
    )


def get_args():
    parser = argparse.ArgumentParser(description="TAP-Out pipeline")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to YAML config (defaults to configs/default.yaml).")
    parser.add_argument("-gt", "--use_gt_rot", action="store_true",
                        help="Use ground-truth camera rotations instead of VGGT-estimated ones.")
    parser.add_argument("-d", "--debugs", action="store_true", help="Emit debug visualisations.")
    parser.add_argument("--split_file", type=str, default=None,
                        help="Optional list of clip names to evaluate (JSON array OR one clip per line).")
    return parser.parse_args()


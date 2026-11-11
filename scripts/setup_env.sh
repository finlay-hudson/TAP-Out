#!/usr/bin/env bash
# One-shot environment bootstrap for TAP-Out.
#
# Order of preference:
#   1. `uv` if installed (recommended — 10× faster resolves).
#   2. Plain `python -m venv` + `pip` otherwise.
#
# Usage:
#   bash scripts/setup_env.sh                                # CUDA 12.8 wheels (default)
#   TORCH_BACKEND=cu121 bash scripts/setup_env.sh            # different CUDA build
#   TORCH_BACKEND=cpu   bash scripts/setup_env.sh            # CPU-only (small)
#   VENV_DIR=/big/disk/.venv bash scripts/setup_env.sh       # venv elsewhere
#   UV_CACHE_DIR=/big/disk/uv_cache bash scripts/setup_env.sh# uv cache elsewhere
#
# Torch cu128 + VGGT + SAM2 + CoTracker unpack to ~20 GB — if `/` is tight,
# set VENV_DIR and UV_CACHE_DIR to a bigger partition before running.

set -euo pipefail

TORCH_BACKEND="${TORCH_BACKEND:-cu128}"
VENV_DIR="${VENV_DIR:-.venv}"

if [[ -n "${UV_CACHE_DIR:-}" ]]; then
    mkdir -p "$UV_CACHE_DIR"
    export UV_CACHE_DIR
fi

echo "==> TAP-Out env bootstrap"
echo "    torch backend : $TORCH_BACKEND"
echo "    venv          : $VENV_DIR"
echo "    uv cache      : ${UV_CACHE_DIR:-<uv default>}"

case "$TORCH_BACKEND" in
    cpu)     TORCH_INDEX="https://download.pytorch.org/whl/cpu" ;;
    cu121)   TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
    cu124)   TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
    cu126)   TORCH_INDEX="https://download.pytorch.org/whl/cu126" ;;
    cu128|*) TORCH_INDEX="https://download.pytorch.org/whl/cu128" ;;
esac

# The PyTorch index publishes wheels with a PEP 440 local-version suffix
# (e.g. "torch-2.8.0+cu128-…whl"). Older uv is strict about matching this,
# so we pin the suffix explicitly. Every supported backend uses "+<backend>".
TORCH_SPEC="torch==2.8.0+${TORCH_BACKEND}"
TV_SPEC="torchvision==0.23.0+${TORCH_BACKEND}"

# VGGT's own metadata pins numpy<2, which fights everything modern. Install
# it separately with --no-deps after the main resolve. Runtime-compatible with
# numpy 2 despite the defensive upstream pin.
VGGT_SPEC="vggt @ git+https://github.com/facebookresearch/vggt.git@44b3afb"

if command -v uv >/dev/null 2>&1; then
    echo "==> Using uv."
    if [[ -d "$VENV_DIR/bin" ]]; then
        echo "    Venv already exists — reusing (delete $VENV_DIR to force rebuild)."
    else
        uv venv "$VENV_DIR" --python 3.11
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    # `uv pip install` is idempotent: if pins are already satisfied it's a fast no-op.
    uv pip install "$TORCH_SPEC" "$TV_SPEC" --index-url "$TORCH_INDEX"
    uv pip install -e .
    uv pip install --no-deps "$VGGT_SPEC"
else
    echo "==> uv not found — falling back to python -m venv + pip."
    echo "    (install uv for a much faster install: https://docs.astral.sh/uv/)"
    if [[ -d "$VENV_DIR/bin" ]]; then
        echo "    Venv already exists — reusing (delete $VENV_DIR to force rebuild)."
    else
        python3 -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install "$TORCH_SPEC" "$TV_SPEC" --index-url "$TORCH_INDEX"
    pip install -e .
    pip install --no-deps "$VGGT_SPEC"
fi

echo
echo "==> Done. Activate with:  source $VENV_DIR/bin/activate"
echo "==> Next:                 bash scripts/download_checkpoints.sh"

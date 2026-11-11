#!/usr/bin/env bash
# Fetch every checkpoint TAP-Out needs into ./checkpoints/.
#
# What this pulls:
#   * SAM2 (Hiera-L)     ~ 0.9 GB — direct URL, always reliable.
#   * Argus SVD U-Net    ~ 3 GB   — Google Drive folder via gdown.
#
# What this does NOT pull (they auto-download on first run):
#   * VGGT     — via HuggingFace, cached in ~/.cache/huggingface.
#   * CoTracker3 — via torch.hub, cached in ~/.cache/torch/hub.
#
# Usage:
#   bash scripts/download_checkpoints.sh              # into ./checkpoints
#   CKPT_DIR=/data/ckpt bash scripts/download_checkpoints.sh  # custom root
#
# Idempotent — files already present are skipped.

set -euo pipefail

CKPT_DIR="${CKPT_DIR:-$(pwd)/checkpoints}"
SAM2_DIR="$CKPT_DIR/sam2"
ARGUS_DIR="$CKPT_DIR/argus"

mkdir -p "$SAM2_DIR" "$ARGUS_DIR"

# ----------------------------- SAM2 ---------------------------------------
SAM2_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
SAM2_PATH="$SAM2_DIR/sam2_hiera_large.pt"

if [[ -s "$SAM2_PATH" ]]; then
    echo "[skip] SAM2 already present: $SAM2_PATH"
else
    echo "[SAM2] downloading $SAM2_URL"
    curl -fL --progress-bar -o "$SAM2_PATH.partial" "$SAM2_URL"
    mv "$SAM2_PATH.partial" "$SAM2_PATH"
    echo "[SAM2] done: $SAM2_PATH"
fi

# ----------------------------- Argus --------------------------------------
# The Argus authors host the SVD U-Net on Google Drive. Folder ID is taken
# from https://github.com/Red-Fairy/argus-code (their inference section).
ARGUS_GDRIVE_ID="1mZpViQY2yvwav-CcxdsQouYu8t52b25G"

ARGUS_WEIGHTS="$ARGUS_DIR/diffusion_pytorch_model.safetensors"

if [[ -s "$ARGUS_WEIGHTS" ]]; then
    echo "[skip] Argus already present: $ARGUS_WEIGHTS"
else
    if ! command -v gdown >/dev/null 2>&1; then
        echo "[Argus] gdown not installed. Installing…"
        pip install --quiet gdown
    fi
    echo "[Argus] downloading Google Drive folder $ARGUS_GDRIVE_ID → $ARGUS_DIR"
    echo "        (~3 GB, may take a while; Google Drive rate-limits anonymous"
    echo "         downloads — see fallback in the README if this fails)"
    # --remaining-ok exists on gdown >= 5.2; older gdown supports the same
    # behaviour by default (best-effort per file).
    GDOWN_FLAGS=(--folder)
    if gdown --help 2>&1 | grep -q -- "--remaining-ok"; then
        GDOWN_FLAGS+=(--remaining-ok)
    fi
    gdown "${GDOWN_FLAGS[@]}" \
          "https://drive.google.com/drive/folders/$ARGUS_GDRIVE_ID" \
          -O "$ARGUS_DIR"
    echo "[Argus] done: $ARGUS_DIR"
fi

# ----------------------------- Summary ------------------------------------
echo
echo "All checkpoints ready under: $CKPT_DIR"
echo "Point configs/default.yaml at these paths (or leave the defaults):"
echo "  paths.sam_checkpoint: $SAM2_PATH"
echo "  paths.unet_path:      $ARGUS_DIR"

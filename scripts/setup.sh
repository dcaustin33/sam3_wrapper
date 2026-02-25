#!/usr/bin/env bash
#
# One-command setup for sam3_wrapper.
#
# This script:
#   1. Installs the Python package and dependencies via uv
#   2. Logs into Hugging Face (if not already authenticated)
#   3. Downloads the SAM 3 model checkpoint
#
# Prerequisites:
#   - uv (https://docs.astral.sh/uv/)
#   - Hugging Face account with access to facebook/sam3
#   - GPU with CUDA support (recommended)
#
# Usage:
#   bash scripts/setup.sh
#   bash scripts/setup.sh --skip-download   # Skip model download
#   bash scripts/setup.sh --dir ./my_ckpts  # Custom checkpoint directory

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SKIP_DOWNLOAD=false
CHECKPOINT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            head -15 "$0" | tail -12
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$PROJECT_DIR"

echo "============================================================"
echo "SAM 3 Wrapper - Setup"
echo "============================================================"

# 1. Install package
echo ""
echo "[1/3] Installing sam3-wrapper and dependencies ..."
if command -v uv &> /dev/null; then
    uv sync
else
    echo "ERROR: uv is not installed. Install it from https://docs.astral.sh/uv/"
    exit 1
fi

# 2. Hugging Face authentication
echo ""
echo "[2/3] Checking Hugging Face authentication ..."
if uv run huggingface-cli whoami &> /dev/null; then
    echo "Already logged in to Hugging Face."
else
    echo "Please log in to Hugging Face (you need access to facebook/sam3):"
    uv run huggingface-cli login
fi

# 3. Download model
if [ "$SKIP_DOWNLOAD" = true ]; then
    echo ""
    echo "[3/3] Skipping model download (--skip-download)"
else
    echo ""
    echo "[3/3] Downloading SAM 3 model ..."
    DIR_FLAG=""
    if [ -n "$CHECKPOINT_DIR" ]; then
        DIR_FLAG="--checkpoint-dir $CHECKPOINT_DIR"
    fi
    uv run sam3-download $DIR_FLAG
fi

echo ""
echo "============================================================"
echo "Setup complete!"
echo ""
echo "Usage:"
echo "  # Python API:"
echo "  from sam3_wrapper import Sam3Segmenter"
echo "  model = Sam3Segmenter()"
echo "  result = model.predict('photo.jpg', text='a person')"
echo ""
echo "  # CLI:"
echo "  uv run sam3-infer --images ./photos --text 'a person' --output ./results"
echo "  uv run sam3-infer --images ./photos --text 'a dog' --alpha  # Save with alpha"
echo "============================================================"

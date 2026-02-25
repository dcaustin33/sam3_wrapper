#!/usr/bin/env bash
#
# Full setup script for sam3_wrapper.
#
# This script:
#   1. Installs the sam3_wrapper package with uv
#   2. Installs the SAM-2 package (outside uv lock to avoid CUDA build issues)
#   3. Downloads SAM2 model checkpoints from Meta's CDN
#   4. Verifies the installation
#
# Prerequisites:
#   - uv (https://docs.astral.sh/uv/)
#   - Python 3.10+
#
# Usage:
#   bash scripts/setup.sh                       # Setup with large SAM2 model (default)
#   bash scripts/setup.sh --variant small        # Use a smaller SAM2 model
#   bash scripts/setup.sh --variant tiny         # Use the smallest SAM2 model
#   bash scripts/setup.sh --all                  # Download all SAM2 variants
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

VARIANT="large"
DOWNLOAD_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        --all)
            DOWNLOAD_ALL=true
            shift
            ;;
        -h|--help)
            head -20 "$0" | tail -16
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "sam3_wrapper - Full Setup"
echo "============================================================"

cd "$PROJECT_DIR"

# Step 1: Install the package and dependencies
echo ""
echo "[1/4] Installing sam3_wrapper with uv ..."
uv sync

# Step 2: Install SAM-2 package (outside lock file to handle CUDA build)
# SAM2_BUILD_CUDA=0 skips the optional CUDA extension that may fail to build.
# The extension only affects small mask artifact removal — masks work fine without it.
echo ""
echo "[2/4] Installing SAM-2 package ..."
if python -c "import sam2" 2>/dev/null; then
    echo "  SAM-2 already installed."
else
    SAM2_BUILD_CUDA=0 uv pip install SAM-2
    echo "  SAM-2 installed."
fi

# Step 3: Download SAM2 checkpoints
echo ""
echo "[3/4] Downloading SAM2 checkpoints ..."
if [ "$DOWNLOAD_ALL" = true ]; then
    uv run sam3-setup --skip-sam2-install --all
else
    uv run sam3-setup --skip-sam2-install --variant "$VARIANT"
fi

# Step 4: Verify installation
echo ""
echo "[4/4] Verifying installation ..."
uv run python -c "
import sys

print('Checking imports ...')

try:
    import torch
    print(f'  torch {torch.__version__}: OK (CUDA: {torch.cuda.is_available()})')
except ImportError as e:
    print(f'  torch: FAILED ({e})')
    sys.exit(1)

try:
    import transformers
    print(f'  transformers {transformers.__version__}: OK')
except ImportError as e:
    print(f'  transformers: FAILED ({e})')
    sys.exit(1)

try:
    import sam2
    print(f'  sam2: OK')
except ImportError as e:
    print(f'  sam2: FAILED ({e})')
    sys.exit(1)

try:
    from sam3_wrapper import Sam3Wrapper
    print(f'  sam3_wrapper: OK')
except ImportError as e:
    print(f'  sam3_wrapper: FAILED ({e})')
    sys.exit(1)

from sam3_wrapper.download import SAM2_VARIANTS, verify_checkpoint
for variant in SAM2_VARIANTS:
    if verify_checkpoint(variant):
        from sam3_wrapper.download import get_checkpoint_path
        path = get_checkpoint_path(variant)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f'  SAM2 {variant}: OK ({size_mb:.0f} MB)')
    else:
        print(f'  SAM2 {variant}: not downloaded')

print()
print('Installation verified successfully!')
"

echo ""
echo "============================================================"
echo "Setup complete!"
echo ""
echo "Quick start:"
echo ""
echo "  # Python API"
echo "  uv run python -c '"
echo "  from sam3_wrapper import Sam3Wrapper"
echo "  model = Sam3Wrapper()"
echo "  result = model.predict(\"photo.jpg\", prompt=\"a person\")"
echo "  '"
echo ""
echo "  # CLI inference"
echo "  uv run sam3-infer --images ./my_images --prompt \"a person\" --output ./results"
echo ""
echo "  # With alpha mask saving"
echo "  uv run sam3-infer --images ./my_images --prompt \"a person\" --output ./results --save-alpha"
echo "============================================================"

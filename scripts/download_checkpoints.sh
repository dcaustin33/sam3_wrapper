#!/usr/bin/env bash
#
# Download SAM 2.1 model checkpoints from Meta's CDN.
#
# Usage:
#   bash scripts/download_checkpoints.sh                       # Download large (default)
#   bash scripts/download_checkpoints.sh --variant small       # Download small variant
#   bash scripts/download_checkpoints.sh --variant tiny        # Download tiny variant
#   bash scripts/download_checkpoints.sh --all                 # Download all variants
#   bash scripts/download_checkpoints.sh --dir ./my_ckpts      # Custom directory
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

VARIANT="large"
DOWNLOAD_ALL=false
CHECKPOINT_DIR=""

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
        --dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            head -12 "$0" | tail -8
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$PROJECT_DIR"

DIR_FLAG=""
if [ -n "$CHECKPOINT_DIR" ]; then
    DIR_FLAG="--checkpoint-dir $CHECKPOINT_DIR"
fi

if [ "$DOWNLOAD_ALL" = true ]; then
    uv run sam3-setup --skip-sam2-install --all $DIR_FLAG
else
    uv run sam3-setup --skip-sam2-install --variant "$VARIANT" $DIR_FLAG
fi

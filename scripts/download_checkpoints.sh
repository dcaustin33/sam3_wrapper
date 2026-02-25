#!/usr/bin/env bash
#
# Download SAM 3 model checkpoint from Hugging Face.
#
# Prerequisites:
#   - huggingface-cli login (authenticate first)
#   - Access approved on the HF model page:
#     https://huggingface.co/facebook/sam3
#
# Usage:
#   bash scripts/download_checkpoints.sh
#   bash scripts/download_checkpoints.sh --dir ./my_ckpts   # Custom directory
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CHECKPOINT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            head -13 "$0" | tail -10
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

uv run sam3-download $DIR_FLAG

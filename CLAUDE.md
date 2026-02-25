# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A pip-installable Python wrapper for text-prompted image segmentation. Combines Grounding DINO (text → bounding boxes) with SAM 2 (boxes → pixel masks). Provide a directory of images and a natural language prompt (e.g. "a man", "a woman"), and get back segmentation masks. Optionally saves masks as the alpha channel of the original images.

## Build & Setup Commands

```bash
# Full one-command setup (install deps, install SAM-2, download checkpoints)
bash scripts/setup.sh

# Step-by-step:
uv sync                                     # Install package and dependencies
SAM2_BUILD_CUDA=0 uv pip install SAM-2      # Install SAM-2 (outside lock file)
uv run sam3-setup --variant large            # Download SAM2 checkpoint
```

## CLI Entry Points

```bash
uv run sam3-setup                          # Install SAM-2 + download checkpoints
uv run sam3-infer --images ./photos --prompt "a person" --output ./results
uv run sam3-infer --images ./photos --prompt "a person" --output ./results --save-alpha
uv run sam3-infer --images ./photos --prompt "a person" --output ./results --save-vis
```

## Running Tests

```bash
uv run pytest            # Requires dev dependencies: uv sync --extra dev
```

## Architecture

The package lives in `src/sam3_wrapper/` with three modules:

- **download.py** — Downloads SAM 2.1 checkpoints from Meta's public CDN. Grounding DINO auto-downloads from HuggingFace on first use. Supports four SAM2 variants: large, base_plus, small, tiny. Environment variable `SAM3_CHECKPOINT_DIR` overrides default `./checkpoints/` directory.

- **inference.py** — Core API. `Sam3Wrapper` class loads Grounding DINO (HuggingFace transformers) and SAM 2 at init time. `predict()` takes one image + text prompt, returns `ImageResult` with `MaskResult` objects containing boolean masks, labels, confidences, and bounding boxes. `predict_batch()` processes a directory. Both support `save_alpha=True` to write PNG files with the mask as the alpha channel.

- **masking.py** — Alpha mask utilities. `save_with_alpha()` saves an image with a mask as its alpha channel (transparent PNG). `apply_alpha_mask()` returns an RGBA numpy array without saving.

## Key Design Patterns

- **Text prompt normalization**: Grounding DINO requires lowercase text ending with a period. The `_normalize_prompt()` function handles this automatically.

- **SAM-2 lives outside uv's lock file**: It is installed via `uv pip install SAM-2` with `SAM2_BUILD_CUDA=0` to skip optional CUDA extension builds. Running `uv sync` will **remove** SAM-2. Reinstall with: `SAM2_BUILD_CUDA=0 uv pip install SAM-2`

- **Grounding DINO auto-downloads**: Models are loaded from HuggingFace via `transformers.AutoModelForZeroShotObjectDetection`. No manual checkpoint download needed — they cache in `~/.cache/huggingface/`.

- **Four SAM2 variants**: large (856MB), base_plus (308MB), small (176MB), tiny (148MB). Variant name maps to checkpoint URL and config in `download.py:SAM2_VARIANTS`.

## Dependencies

Uses `uv` as package manager with `hatchling` build backend. PyTorch and TorchVision are pinned to CUDA 12.4 index via `tool.uv.sources` (adjust URL for other CUDA versions or CPU).

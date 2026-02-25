# sam3_wrapper

A pip-installable wrapper around Meta's **SAM 3 (Segment Anything with Concepts)** for text-prompted image segmentation.

Give it a directory of images and a text prompt (e.g. "a man", "a dog") and get back binary masks for every matching object. Optionally save the masks as alpha channels on the original images.

## Features

- **Text-prompted segmentation** — describe what you want to segment in plain English
- **Batch processing** — point at a directory and process all images at once
- **Alpha channel output** — save masked images as RGBA PNGs with transparent backgrounds
- **Simple Python API** — `Sam3Segmenter` class for use from other packages
- **CLI tools** — `sam3-infer`, `sam3-download`, `sam3-setup`
- **Easy setup** — one script handles everything including model download

## Prerequisites

- Python 3.12+
- `uv` package manager
- GPU with CUDA support
- Hugging Face account with access to [facebook/sam3](https://huggingface.co/facebook/sam3)

## Quick Start

**One-command setup:**

```bash
git clone <repo-url> && cd sam3_wrapper
bash scripts/setup.sh
```

**Step-by-step:**

```bash
uv sync
huggingface-cli login
uv run sam3-download
```

## Usage

### Python API

```python
from sam3_wrapper import Sam3Segmenter

model = Sam3Segmenter()

# Single image
result = model.predict("photo.jpg", text="a person")
print(f"Found {result.num_masks} masks")
for mask in result.masks:
    print(mask.shape)  # (H, W) boolean array

# Batch processing
results = model.predict_directory(
    "./images/",
    text="a dog",
    output_dir="./output/",
    save_masks=True,
    save_alpha=True,  # Save with transparent background
)
```

### CLI

```bash
# Basic segmentation
uv run sam3-infer --images ./photos --text "a person" --output ./results

# Save with alpha channel (transparent background)
uv run sam3-infer --images ./photos --text "a cat" --output ./results --alpha

# Custom thresholds
uv run sam3-infer --images ./photos --text "a car" --threshold 0.3 --mask-threshold 0.4
```

### CLI Commands

| Command | Purpose |
|---------|---------|
| `sam3-setup` | Download model from Hugging Face |
| `sam3-download` | Download model checkpoint |
| `sam3-infer` | Run text-prompted segmentation on images |

## Environment Variables

- `SAM3_CHECKPOINT_DIR`: Override checkpoint storage path (default: `<project>/checkpoints/`)

## Using as a Dependency

Install from git in another project:

```bash
uv add sam3-wrapper --git <repo-url>
```

Then use the Python API:

```python
from sam3_wrapper import Sam3Segmenter

model = Sam3Segmenter()
result = model.predict(image, text="a person")
```

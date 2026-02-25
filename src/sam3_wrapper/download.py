"""
Download SAM 3 model from Hugging Face.

The model is gated and requires access approval at:
  https://huggingface.co/facebook/sam3

You must first:
  1. Request access on the Hugging Face model page
  2. Authenticate via `huggingface-cli login`
"""

import argparse
import os
from pathlib import Path

HF_REPO_ID = "facebook/sam3"

_DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"


def get_checkpoint_dir() -> Path:
    """Return the checkpoint directory, respecting SAM3_CHECKPOINT_DIR env var."""
    env = os.environ.get("SAM3_CHECKPOINT_DIR")
    if env:
        return Path(env)
    return _DEFAULT_CHECKPOINT_DIR


def get_checkpoint_path() -> Path:
    """Return the expected checkpoint directory for the SAM3 model."""
    return get_checkpoint_dir() / "sam3"


def download_checkpoint(checkpoint_dir: Path | None = None) -> Path:
    """
    Download the SAM 3 model from Hugging Face.

    Args:
        checkpoint_dir: Override directory for storing checkpoints.

    Returns:
        Path to the downloaded checkpoint directory.
    """
    from huggingface_hub import snapshot_download

    checkpoint_dir = checkpoint_dir or get_checkpoint_dir()
    local_dir = checkpoint_dir / "sam3"

    print(f"Downloading {HF_REPO_ID} to {local_dir} ...")
    print("(You must have requested access and be logged in via `huggingface-cli login`)")

    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=str(local_dir),
    )

    print(f"Download complete: {local_dir}")
    _print_checkpoint_contents(local_dir)
    return local_dir


def _print_checkpoint_contents(path: Path) -> None:
    """Print a summary of downloaded checkpoint files."""
    print("\nDownloaded files:")
    for f in sorted(path.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            rel = f.relative_to(path)
            print(f"  {rel} ({size_mb:.1f} MB)")


def verify_checkpoint() -> bool:
    """Check whether the SAM3 model has been downloaded."""
    ckpt_dir = get_checkpoint_path()
    # Check for either safetensors or pt checkpoint
    safetensors = ckpt_dir / "model.safetensors"
    pt_ckpt = ckpt_dir / "sam3.pt"
    config = ckpt_dir / "config.json"
    return config.exists() and (safetensors.exists() or pt_ckpt.exists())


def cli_download() -> None:
    """CLI entry point: sam3-download"""
    parser = argparse.ArgumentParser(
        description="Download SAM 3 model checkpoint from Hugging Face."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory to store checkpoints (default: <package_root>/checkpoints/)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SAM 3 - Model Download")
    print("=" * 60)

    download_checkpoint(args.checkpoint_dir)

    print("\n" + "=" * 60)
    print("Done! You can now run inference:")
    print("  sam3-infer --images ./my_images --text 'a person' --output ./results")
    print("=" * 60)


def cli_setup() -> None:
    """CLI entry point: sam3-setup — download the model."""
    parser = argparse.ArgumentParser(
        description="Set up SAM 3: download model from Hugging Face."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory to store checkpoints (default: <package_root>/checkpoints/)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SAM 3 Wrapper - Setup")
    print("=" * 60)

    print("\n[1/1] Downloading SAM 3 model from Hugging Face ...")
    download_checkpoint(args.checkpoint_dir)

    print("\n" + "=" * 60)
    print("Setup complete!")
    print()
    print("Next steps:")
    print("  Run inference:  sam3-infer --images ./my_images --text 'a person'")
    print("=" * 60)

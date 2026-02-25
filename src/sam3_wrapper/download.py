"""
Download SAM 2.1 model checkpoints from Meta's public CDN.

Grounding DINO models are loaded from HuggingFace and download automatically
on first use — no manual download needed for those.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

SAM2_VARIANTS = {
    "large": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "config": "sam2.1/sam2.1_hiera_l.yaml",
        "filename": "sam2.1_hiera_large.pt",
        "description": "SAM 2.1 Hiera Large (856 MB)",
    },
    "base_plus": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "config": "sam2.1/sam2.1_hiera_b+.yaml",
        "filename": "sam2.1_hiera_base_plus.pt",
        "description": "SAM 2.1 Hiera Base+ (308 MB)",
    },
    "small": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "config": "sam2.1/sam2.1_hiera_s.yaml",
        "filename": "sam2.1_hiera_small.pt",
        "description": "SAM 2.1 Hiera Small (176 MB)",
    },
    "tiny": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "config": "sam2.1/sam2.1_hiera_t.yaml",
        "filename": "sam2.1_hiera_tiny.pt",
        "description": "SAM 2.1 Hiera Tiny (148 MB)",
    },
}

_DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"


def get_checkpoint_dir() -> Path:
    """Return the checkpoint directory, respecting SAM3_CHECKPOINT_DIR env var."""
    env = os.environ.get("SAM3_CHECKPOINT_DIR")
    if env:
        return Path(env)
    return _DEFAULT_CHECKPOINT_DIR


def get_checkpoint_path(variant: str = "large") -> Path:
    """Return the expected checkpoint file path for a given SAM2 variant."""
    info = SAM2_VARIANTS.get(variant)
    if info is None:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from: {list(SAM2_VARIANTS.keys())}"
        )
    return get_checkpoint_dir() / info["filename"]


def get_sam2_config(variant: str = "large") -> str:
    """Return the SAM2 model config name for a given variant."""
    info = SAM2_VARIANTS.get(variant)
    if info is None:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from: {list(SAM2_VARIANTS.keys())}"
        )
    return info["config"]


def verify_checkpoint(variant: str = "large") -> bool:
    """Check whether a SAM2 checkpoint has been downloaded."""
    return get_checkpoint_path(variant).exists()


def download_checkpoint(variant: str = "large", checkpoint_dir: Path | None = None) -> Path:
    """
    Download a SAM 2.1 checkpoint from Meta's CDN.

    Args:
        variant: Model size — "large", "base_plus", "small", or "tiny".
        checkpoint_dir: Override directory for storing checkpoints.

    Returns:
        Path to the downloaded checkpoint file.
    """
    info = SAM2_VARIANTS.get(variant)
    if info is None:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from: {list(SAM2_VARIANTS.keys())}"
        )

    checkpoint_dir = checkpoint_dir or get_checkpoint_dir()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_path = checkpoint_dir / info["filename"]

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Checkpoint already exists: {output_path} ({size_mb:.0f} MB)")
        return output_path

    print(f"Downloading {info['description']} ...")
    print(f"  URL: {info['url']}")
    print(f"  Destination: {output_path}")

    import urllib.request

    urllib.request.urlretrieve(info["url"], str(output_path))

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Download complete: {output_path} ({size_mb:.0f} MB)")
    return output_path


def install_sam2() -> None:
    """Install the SAM-2 package from PyPI."""
    import shutil

    uv = shutil.which("uv")
    print("Installing SAM-2 package ...")
    if uv:
        subprocess.check_call([uv, "pip", "install", "SAM-2"])
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "SAM-2"])
    print("SAM-2 installed.")


def ensure_sam2() -> None:
    """Ensure SAM-2 is installed. Installs automatically if missing."""
    try:
        import sam2  # noqa: F401
    except ImportError:
        print("SAM-2 not found, installing automatically ...")
        install_sam2()


def cli_setup() -> None:
    """CLI entry point: sam3-setup — download checkpoints and install SAM-2."""
    parser = argparse.ArgumentParser(
        description="Set up sam3_wrapper: install SAM-2 and download checkpoints."
    )
    parser.add_argument(
        "--variant",
        choices=list(SAM2_VARIANTS.keys()),
        default="large",
        help="SAM2 model variant to download (default: large)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory to store checkpoints (default: <project>/checkpoints/)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="download_all",
        help="Download all SAM2 model variants",
    )
    parser.add_argument(
        "--skip-sam2-install",
        action="store_true",
        help="Skip installing the SAM-2 package",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("sam3_wrapper - Setup")
    print("=" * 60)

    # Step 1: Install SAM-2
    if not args.skip_sam2_install:
        print("\n[1/2] Installing SAM-2 package ...")
        ensure_sam2()
    else:
        print("\n[1/2] Skipping SAM-2 installation")

    # Step 2: Download checkpoints
    print("\n[2/2] Downloading SAM2 checkpoints ...")
    if args.download_all:
        for variant in SAM2_VARIANTS:
            print(f"\n--- {variant} ---")
            download_checkpoint(variant, args.checkpoint_dir)
    else:
        download_checkpoint(args.variant, args.checkpoint_dir)

    print("\n" + "=" * 60)
    print("Setup complete!")
    print()
    print("Quick start:")
    print('  uv run sam3-infer --images ./photos --prompt "a person" --output ./results')
    print("=" * 60)

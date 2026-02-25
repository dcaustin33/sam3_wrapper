"""
High-level inference API for SAM 3 text-prompted segmentation.

Provides Sam3Segmenter class for single-image and batch inference,
using text prompts to segment objects in images.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam3_wrapper.download import HF_REPO_ID, get_checkpoint_path, verify_checkpoint

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}


@dataclass
class SegmentationResult:
    """Result for a single image's text-prompted segmentation."""

    image_path: str
    text_prompt: str
    masks: list[np.ndarray]
    boxes: list[np.ndarray]
    scores: list[float]
    image: Image.Image | None = field(default=None, repr=False)

    @property
    def num_masks(self) -> int:
        return len(self.masks)


def _collect_images(path: str | Path) -> list[Path]:
    """Collect image files from a path (file or directory)."""
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = []
        for f in sorted(path.iterdir()):
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                files.append(f)
        return files
    raise FileNotFoundError(f"Path does not exist: {path}")


def _apply_mask_as_alpha(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Apply a binary mask as the alpha channel of an image.

    Args:
        image: RGB PIL Image.
        mask: Boolean or uint8 mask array (H, W).

    Returns:
        RGBA PIL Image where the alpha channel is the mask.
    """
    rgba = image.convert("RGBA")
    alpha = (mask.astype(np.uint8) * 255)
    alpha_img = Image.fromarray(alpha, mode="L")
    rgba.putalpha(alpha_img)
    return rgba


class Sam3Segmenter:
    """
    High-level wrapper for SAM 3 text-prompted image segmentation.

    Example:
        from sam3_wrapper import Sam3Segmenter

        model = Sam3Segmenter()

        # Single image
        result = model.predict("photo.jpg", text="a person")
        for mask in result.masks:
            print(mask.shape)

        # Batch of images
        results = model.predict_directory(
            "./images/", text="a dog", output_dir="./output/"
        )

    Args:
        device: PyTorch device string. Defaults to "cuda" if available.
        threshold: Detection confidence threshold (default: 0.5).
        mask_threshold: Mask binarization threshold (default: 0.5).
        checkpoint_path: Path to a local checkpoint directory. If None,
            uses the default HuggingFace cache or local checkpoint dir.
    """

    def __init__(
        self,
        device: str | None = None,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        checkpoint_path: str | None = None,
    ):
        from transformers import Sam3Model, Sam3Processor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.mask_threshold = mask_threshold

        # Determine model source: local checkpoint or HuggingFace hub
        model_id = checkpoint_path or str(get_checkpoint_path())
        if not Path(model_id).exists():
            # Fall back to HuggingFace hub download
            model_id = HF_REPO_ID

        print(f"Loading SAM 3 model from {model_id} ...")
        self._model = Sam3Model.from_pretrained(model_id).to(self.device)
        self._processor = Sam3Processor.from_pretrained(model_id)
        print("Model loaded.")

    def predict(
        self,
        image: str | Path | Image.Image,
        text: str,
        threshold: float | None = None,
        mask_threshold: float | None = None,
    ) -> SegmentationResult:
        """
        Run text-prompted segmentation on a single image.

        Args:
            image: Path to an image file, or a PIL Image.
            text: Text prompt describing the object to segment (e.g. "a man").
            threshold: Override the default detection confidence threshold.
            mask_threshold: Override the default mask binarization threshold.

        Returns:
            SegmentationResult with masks, boxes, and scores.
        """
        thr = threshold if threshold is not None else self.threshold
        mask_thr = mask_threshold if mask_threshold is not None else self.mask_threshold

        if isinstance(image, (str, Path)):
            image_path = str(image)
            pil_image = Image.open(image_path).convert("RGB")
        else:
            image_path = "<PIL.Image>"
            pil_image = image.convert("RGB")

        inputs = self._processor(
            images=pil_image,
            text=text,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=thr,
            mask_threshold=mask_thr,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        masks = [m.cpu().numpy() for m in results["masks"]]
        boxes = [b.cpu().numpy() for b in results["boxes"]]
        scores = [s.item() for s in results["scores"]]

        return SegmentationResult(
            image_path=image_path,
            text_prompt=text,
            masks=masks,
            boxes=boxes,
            scores=scores,
            image=pil_image,
        )

    def predict_directory(
        self,
        images_path: str | Path,
        text: str,
        output_dir: str | Path | None = None,
        threshold: float | None = None,
        mask_threshold: float | None = None,
        save_masks: bool = True,
        save_alpha: bool = False,
    ) -> list[SegmentationResult]:
        """
        Run text-prompted segmentation on a directory of images.

        Args:
            images_path: Directory or single file path.
            text: Text prompt describing the object to segment.
            output_dir: If set, save mask images here.
            threshold: Override detection confidence threshold.
            mask_threshold: Override mask binarization threshold.
            save_masks: Save binary mask images to output_dir.
            save_alpha: Save the input image with the mask applied as
                the alpha channel (RGBA PNG).

        Returns:
            List of SegmentationResult, one per input image.
        """
        from tqdm import tqdm

        image_files = _collect_images(images_path)
        if not image_files:
            print(f"No images found at {images_path}")
            return []

        print(f"Processing {len(image_files)} images with prompt: '{text}' ...")

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for img_path in tqdm(image_files, desc="Segmenting"):
            result = self.predict(
                img_path,
                text=text,
                threshold=threshold,
                mask_threshold=mask_threshold,
            )
            results.append(result)

            if output_dir is not None and result.num_masks > 0:
                self._save_results(result, output_dir, img_path.stem, save_masks, save_alpha)

        total_masks = sum(r.num_masks for r in results)
        print(f"Done: {len(results)} images, {total_masks} masks found.")
        return results

    @staticmethod
    def _save_results(
        result: SegmentationResult,
        output_dir: Path,
        image_name: str,
        save_masks: bool,
        save_alpha: bool,
    ) -> None:
        """Save mask outputs for a single image."""
        if save_masks:
            # Combine all masks into a single mask image
            if result.masks:
                combined = np.zeros_like(result.masks[0], dtype=np.uint8)
                for i, mask in enumerate(result.masks):
                    combined[mask > 0] = (i + 1) * (255 // max(len(result.masks), 1))
                mask_img = Image.fromarray(combined, mode="L")
                mask_img.save(output_dir / f"{image_name}_mask.png")

            # Also save individual masks
            for i, mask in enumerate(result.masks):
                mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
                mask_img.save(output_dir / f"{image_name}_mask_{i}.png")

        if save_alpha and result.image is not None:
            # Union of all masks as alpha
            if result.masks:
                union_mask = np.zeros_like(result.masks[0], dtype=bool)
                for mask in result.masks:
                    union_mask |= mask.astype(bool)
                alpha_img = _apply_mask_as_alpha(result.image, union_mask)
                alpha_img.save(output_dir / f"{image_name}_alpha.png")


def cli_infer() -> None:
    """CLI entry point: sam3-infer"""
    parser = argparse.ArgumentParser(
        description="Run SAM 3 text-prompted segmentation on images."
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to an image file or directory of images",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text prompt describing the object to segment (e.g. 'a person')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./sam3_output",
        help="Output directory for results (default: ./sam3_output)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: cuda if available)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Mask binarization threshold (default: 0.5)",
    )
    parser.add_argument(
        "--no-masks",
        action="store_true",
        help="Skip saving binary mask images",
    )
    parser.add_argument(
        "--alpha",
        action="store_true",
        help="Save input images with mask as alpha channel (RGBA PNG)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to local model checkpoint directory",
    )
    args = parser.parse_args()

    model = Sam3Segmenter(
        device=args.device,
        threshold=args.threshold,
        mask_threshold=args.mask_threshold,
        checkpoint_path=args.checkpoint_path,
    )

    model.predict_directory(
        images_path=args.images,
        text=args.text,
        output_dir=args.output,
        save_masks=not args.no_masks,
        save_alpha=args.alpha,
    )

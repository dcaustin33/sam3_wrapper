"""
High-level inference API for SAM 3 text-prompted segmentation.

Provides Sam3Segmenter class for single-image and batch inference,
using text prompts to segment objects in images.
"""

import argparse
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue

import cv2
import numpy as np
import torch
from PIL import Image

from sam3_wrapper.download import HF_REPO_ID, get_checkpoint_path, verify_checkpoint

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}


@dataclass
class SegmentationResult:
    """Result for a single image's text-prompted segmentation."""

    image_path: str
    text_prompt: str
    masks: list[np.ndarray]
    boxes: list[np.ndarray]
    scores: list[float]
    image: Image.Image | None = field(default=None, repr=False)  # may be RGB or RGBA

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


def _erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Erode a binary mask inward by a given number of pixels."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1))
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return eroded.astype(mask.dtype)


def _apply_mask_as_alpha(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Apply a binary mask as the alpha channel of an image.

    Any existing alpha channel on the input image is discarded and
    replaced by the mask. The RGB pixel data is preserved as-is.

    Args:
        image: PIL Image (RGB or RGBA). If RGBA, the existing alpha is
            stripped and overridden by the mask.
        mask: Boolean or uint8 mask array (H, W).

    Returns:
        RGBA PIL Image where the alpha channel is the mask.
    """
    # Strip any existing alpha so we start from clean RGB data
    rgb = image.convert("RGB")
    rgba = rgb.convert("RGBA")
    alpha = (mask.astype(np.uint8) * 255)
    alpha_img = Image.fromarray(alpha, mode="L")
    rgba.putalpha(alpha_img)
    return rgba


class _BackgroundSaver:
    """Offloads image encoding and disk I/O to a pool of daemon worker threads.

    Uses a bounded queue so the main (GPU) thread is not blocked by PNG
    encoding while still applying back-pressure if saves fall behind.
    """

    def __init__(self, num_workers: int = 4, maxsize: int = 100):
        self._queue: Queue = Queue(maxsize=maxsize)
        self._threads = []
        for _ in range(num_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self._threads.append(t)

    def _worker(self) -> None:
        while True:
            fn = self._queue.get()
            if fn is None:
                self._queue.task_done()
                break
            try:
                fn()
            except Exception:
                logger.exception("Background image save failed")
            self._queue.task_done()

    def submit(self, fn) -> None:
        self._queue.put(fn)

    def flush(self) -> None:
        """Block until all queued saves complete."""
        self._queue.join()

    def shutdown(self) -> None:
        """Stop all worker threads. Safe to call multiple times."""
        for _ in self._threads:
            self._queue.put(None)
        for t in self._threads:
            t.join(timeout=60)
        self._threads.clear()


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
            image: Path to an image file, or a PIL Image (RGB or RGBA).
                If the image has an alpha channel it is stripped for inference
                but preserved on the result for alpha-aware saving.
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
            pil_image = Image.open(image_path)
        else:
            image_path = "<PIL.Image>"
            pil_image = image

        # The model needs RGB; keep the original for alpha-aware saving
        rgb_image = pil_image.convert("RGB")

        inputs = self._processor(
            images=rgb_image,
            text=text,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad(), torch.autocast(self.device):
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
            image=pil_image,  # preserves original mode (RGB or RGBA)
        )

    def predict_batch(
        self,
        images: list[str | Path | Image.Image],
        text: str,
        threshold: float | None = None,
        mask_threshold: float | None = None,
    ) -> list[SegmentationResult]:
        """
        Run text-prompted segmentation on a batch of images in a single forward pass.

        Args:
            images: List of image paths or PIL Images.
            text: Text prompt describing the object to segment.
            threshold: Override detection confidence threshold.
            mask_threshold: Override mask binarization threshold.

        Returns:
            List of SegmentationResult, one per input image.
        """
        thr = threshold if threshold is not None else self.threshold
        mask_thr = mask_threshold if mask_threshold is not None else self.mask_threshold

        pil_images = []
        image_paths = []
        for img in images:
            if isinstance(img, (str, Path)):
                image_paths.append(str(img))
                pil_images.append(Image.open(img))
            else:
                image_paths.append("<PIL.Image>")
                pil_images.append(img)

        rgb_images = [img.convert("RGB") for img in pil_images]
        texts = [text] * len(rgb_images)

        inputs = self._processor(
            images=rgb_images,
            text=texts,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad(), torch.autocast(self.device):
            outputs = self._model(**inputs)

        batch_results = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=thr,
            mask_threshold=mask_thr,
            target_sizes=inputs.get("original_sizes").tolist(),
        )

        results = []
        for i, res in enumerate(batch_results):
            masks = [m.cpu().numpy() for m in res["masks"]]
            boxes = [b.cpu().numpy() for b in res["boxes"]]
            scores = [s.item() for s in res["scores"]]
            results.append(SegmentationResult(
                image_path=image_paths[i],
                text_prompt=text,
                masks=masks,
                boxes=boxes,
                scores=scores,
                image=pil_images[i],
            ))

        return results

    def predict_directory(
        self,
        images_path: str | Path,
        text: str,
        output_dir: str | Path | None = None,
        threshold: float | None = None,
        mask_threshold: float | None = None,
        save_masks: bool = True,
        save_alpha: bool = False,
        erode_pixels: int = 0,
        batch_size: int = 1,
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
            erode_pixels: Shrink masks inward by this many pixels (default: 0).
            batch_size: Number of images to process per forward pass (default: 1).

        Returns:
            List of SegmentationResult, one per input image.
        """
        from tqdm import tqdm

        image_files = _collect_images(images_path)
        if not image_files:
            print(f"No images found at {images_path}")
            return []

        print(f"Processing {len(image_files)} images with prompt: '{text}' (batch_size={batch_size}) ...")

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        bg_saver = _BackgroundSaver() if output_dir is not None else None

        results = []
        batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]
        pbar = tqdm(total=len(image_files), desc="Segmenting")
        for batch in batches:
            if len(batch) == 1:
                batch_results = [self.predict(
                    batch[0], text=text, threshold=threshold, mask_threshold=mask_threshold,
                )]
            else:
                batch_results = self.predict_batch(
                    batch, text=text, threshold=threshold, mask_threshold=mask_threshold,
                )
            for result, img_path in zip(batch_results, batch):
                results.append(result)
                if output_dir is not None and result.num_masks > 0:
                    bg_saver.submit(lambda r=result, d=output_dir, n=img_path.stem: (
                        self._save_results(r, d, n, save_masks, save_alpha, erode_pixels)
                    ))
            pbar.update(len(batch))
        pbar.close()

        if bg_saver is not None:
            bg_saver.flush()
            bg_saver.shutdown()

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
        erode_pixels: int = 0,
    ) -> None:
        """Save mask outputs for a single image."""
        masks = result.masks
        if erode_pixels > 0:
            masks = [_erode_mask(m, erode_pixels) for m in masks]

        if save_alpha and result.image is not None and masks:
            # Save only the RGBA image with the original filename
            union_mask = np.zeros_like(masks[0], dtype=bool)
            for mask in masks:
                union_mask |= mask.astype(bool)
            alpha_img = _apply_mask_as_alpha(result.image, union_mask)
            alpha_img.save(output_dir / f"{image_name}.png")
            return

        if save_masks:
            # Combine all masks into a single mask image
            if masks:
                combined = np.zeros_like(masks[0], dtype=np.uint8)
                for i, mask in enumerate(masks):
                    combined[mask > 0] = (i + 1) * (255 // max(len(masks), 1))
                mask_img = Image.fromarray(combined, mode="L")
                mask_img.save(output_dir / f"{image_name}_mask.png")

            # Also save individual masks
            for i, mask in enumerate(masks):
                mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
                mask_img.save(output_dir / f"{image_name}_mask_{i}.png")


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
        "--erode-pixels",
        type=int,
        default=0,
        help="Shrink masks inward by this many pixels (default: 0)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to local model checkpoint directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of images per forward pass (default: 1)",
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
        erode_pixels=args.erode_pixels,
        batch_size=args.batch_size,
    )

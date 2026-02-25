"""
High-level inference API for text-prompted image segmentation.

Combines Grounding DINO (text → bounding boxes) with SAM 2 (boxes → masks)
to produce segmentation masks from natural language prompts.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sam3_wrapper.download import (
    SAM2_VARIANTS,
    ensure_sam2,
    get_checkpoint_path,
    get_sam2_config,
    verify_checkpoint,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}


@dataclass
class MaskResult:
    """Result for a single detected object mask."""

    mask: np.ndarray  # (H, W) boolean mask
    label: str
    confidence: float
    bbox: np.ndarray  # (4,) xyxy format


@dataclass
class ImageResult:
    """Result for a single image containing zero or more detected objects."""

    image_path: str
    masks: list[MaskResult]
    image_rgb: np.ndarray | None = field(default=None, repr=False)

    @property
    def num_masks(self) -> int:
        return len(self.masks)

    @property
    def combined_mask(self) -> np.ndarray:
        """Combine all masks into a single boolean mask (logical OR)."""
        if not self.masks:
            if self.image_rgb is not None:
                h, w = self.image_rgb.shape[:2]
                return np.zeros((h, w), dtype=bool)
            return np.zeros((0, 0), dtype=bool)
        combined = np.zeros_like(self.masks[0].mask, dtype=bool)
        for m in self.masks:
            combined |= m.mask
        return combined


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


def _normalize_prompt(prompt: str) -> str:
    """Normalize a text prompt for Grounding DINO.

    Grounding DINO requires text queries to be lowercase and each
    class/phrase to end with a period.
    """
    prompt = prompt.strip().lower()
    if not prompt.endswith("."):
        prompt += "."
    return prompt


class Sam3Wrapper:
    """
    Text-prompted image segmentation using Grounding DINO + SAM 2.

    Given an image and a text prompt (e.g. "a man", "a dog"), detects
    objects matching the prompt and produces pixel-level segmentation masks.

    Example:
        from sam3_wrapper import Sam3Wrapper

        model = Sam3Wrapper()

        # Single image
        result = model.predict("photo.jpg", prompt="a person")
        for mask_result in result.masks:
            print(mask_result.label, mask_result.mask.shape)

        # Save with alpha mask
        result = model.predict(
            "photo.jpg",
            prompt="a person",
            save_alpha=True,
            output_dir="./output",
        )

        # Batch of images
        results = model.predict_batch(
            "./images/",
            prompt="a dog",
            output_dir="./output/",
        )

    Args:
        sam2_variant: SAM2 model size — "large", "base_plus", "small", or "tiny".
        grounding_model: HuggingFace model ID for Grounding DINO.
        device: PyTorch device string. Defaults to "cuda" if available, else "cpu".
        box_threshold: Grounding DINO box confidence threshold.
        text_threshold: Grounding DINO text-box association threshold.
    """

    def __init__(
        self,
        sam2_variant: str = "large",
        grounding_model: str = "IDEA-Research/grounding-dino-tiny",
        device: str | None = None,
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
    ):
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.sam2_variant = sam2_variant

        # Ensure SAM-2 package is installed
        ensure_sam2()

        # Verify checkpoint exists
        if not verify_checkpoint(sam2_variant):
            raise FileNotFoundError(
                f"SAM2 checkpoint not found for variant '{sam2_variant}'. "
                f"Run `sam3-setup --variant {sam2_variant}` or "
                f"`bash scripts/setup.sh` to download it."
            )

        # Enable bfloat16 for performance
        torch.autocast(device_type=self.device, dtype=torch.bfloat16).__enter__()
        if (
            torch.cuda.is_available()
            and torch.cuda.get_device_properties(0).major >= 8
        ):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Build SAM 2 predictor
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        checkpoint_path = str(get_checkpoint_path(sam2_variant))
        model_config = get_sam2_config(sam2_variant)
        sam2_model = build_sam2(model_config, checkpoint_path, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        # Build Grounding DINO from HuggingFace (auto-downloads on first use)
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(grounding_model)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            grounding_model
        ).to(self.device)

    def predict(
        self,
        image: str | Path | np.ndarray,
        prompt: str,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
        save_alpha: bool = False,
        output_dir: str | Path | None = None,
    ) -> ImageResult:
        """
        Run text-prompted segmentation on a single image.

        Args:
            image: Path to an image file, or an RGB numpy array (HxWx3 uint8).
            prompt: Text description of what to segment (e.g. "a man", "a dog").
            box_threshold: Override Grounding DINO box confidence threshold.
            text_threshold: Override Grounding DINO text threshold.
            save_alpha: If True, save the image with the combined mask as its
                alpha channel (PNG format) to output_dir.
            output_dir: Directory for saving alpha-masked images. Required if
                save_alpha is True.

        Returns:
            ImageResult with detected masks, labels, and bounding boxes.
        """
        import torch
        from PIL import Image

        box_thr = box_threshold if box_threshold is not None else self.box_threshold
        text_thr = text_threshold if text_threshold is not None else self.text_threshold

        # Load image
        if isinstance(image, (str, Path)):
            image_path = str(image)
            pil_image = Image.open(image_path).convert("RGB")
            image_rgb = np.array(pil_image)
        else:
            image_path = "<array>"
            image_rgb = image
            pil_image = Image.fromarray(image_rgb)

        # Normalize prompt for Grounding DINO
        text = _normalize_prompt(prompt)

        # Set image on SAM 2
        self.sam2_predictor.set_image(image_rgb)

        # Run Grounding DINO
        inputs = self.processor(
            images=pil_image, text=text, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_thr,
            text_threshold=text_thr,
            target_sizes=[pil_image.size[::-1]],  # (height, width)
        )

        input_boxes = results[0]["boxes"].cpu().numpy()
        confidences = results[0]["scores"].cpu().numpy().tolist()
        labels = results[0]["labels"]

        # Run SAM 2 with box prompts
        mask_results = []
        if len(input_boxes) > 0:
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )

            # Normalize mask shape to (N, H, W)
            if masks.ndim == 4:
                masks = masks.squeeze(1)

            for i in range(len(masks)):
                mask_results.append(
                    MaskResult(
                        mask=masks[i].astype(bool),
                        label=labels[i],
                        confidence=confidences[i],
                        bbox=input_boxes[i],
                    )
                )

        result = ImageResult(
            image_path=image_path,
            masks=mask_results,
            image_rgb=image_rgb,
        )

        # Save alpha-masked image if requested
        if save_alpha and result.num_masks > 0:
            from sam3_wrapper.masking import save_with_alpha

            if output_dir is None:
                raise ValueError("output_dir is required when save_alpha=True")
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            stem = Path(image_path).stem if image_path != "<array>" else "output"
            alpha_path = output_dir / f"{stem}_alpha.png"
            save_with_alpha(image_rgb, result.combined_mask, alpha_path)

        return result

    def predict_batch(
        self,
        images_path: str | Path,
        prompt: str,
        output_dir: str | Path | None = None,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
        save_alpha: bool = False,
        save_visualization: bool = False,
    ) -> list[ImageResult]:
        """
        Run text-prompted segmentation on a directory of images.

        Args:
            images_path: Directory or single file path.
            prompt: Text description of what to segment.
            output_dir: If set, save results here.
            box_threshold: Override Grounding DINO box confidence threshold.
            text_threshold: Override Grounding DINO text threshold.
            save_alpha: Save images with masks as alpha channels.
            save_visualization: Save annotated visualization images.

        Returns:
            List of ImageResult, one per input image.
        """
        from tqdm import tqdm

        image_files = _collect_images(images_path)
        if not image_files:
            print(f"No images found at {images_path}")
            return []

        print(f"Processing {len(image_files)} images with prompt: '{prompt}' ...")

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for img_path in tqdm(image_files, desc="Inference"):
            result = self.predict(
                img_path,
                prompt=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                save_alpha=save_alpha,
                output_dir=str(output_dir) if output_dir else None,
            )
            results.append(result)

            if save_visualization and output_dir is not None and result.num_masks > 0:
                self._save_visualization(result, output_dir, img_path.stem)

        total_masks = sum(r.num_masks for r in results)
        print(f"Done: {len(results)} images, {total_masks} objects detected.")
        return results

    def _save_visualization(
        self, result: ImageResult, output_dir: Path, image_name: str
    ) -> None:
        """Save an annotated visualization image."""
        import cv2
        import supervision as sv

        if result.image_rgb is None or result.num_masks == 0:
            return

        img_bgr = cv2.cvtColor(result.image_rgb, cv2.COLOR_RGB2BGR)
        masks_array = np.array([m.mask for m in result.masks])
        boxes_array = np.array([m.bbox for m in result.masks])
        class_ids = np.arange(len(result.masks))

        detections = sv.Detections(
            xyxy=boxes_array,
            mask=masks_array,
            class_id=class_ids,
        )

        labels = [
            f"{m.label} {m.confidence:.2f}" for m in result.masks
        ]

        annotated = img_bgr.copy()
        annotated = sv.BoxAnnotator().annotate(scene=annotated, detections=detections)
        annotated = sv.LabelAnnotator().annotate(
            scene=annotated, detections=detections, labels=labels
        )
        annotated = sv.MaskAnnotator().annotate(scene=annotated, detections=detections)

        out_path = output_dir / f"{image_name}_vis.jpg"
        cv2.imwrite(str(out_path), annotated)


def cli_infer() -> None:
    """CLI entry point: sam3-infer"""
    parser = argparse.ArgumentParser(
        description="Run text-prompted image segmentation on images."
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to an image file or directory of images",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help='Text prompt describing what to segment (e.g. "a person", "a dog")',
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./sam3_output",
        help="Output directory for results (default: ./sam3_output)",
    )
    parser.add_argument(
        "--variant",
        choices=list(SAM2_VARIANTS.keys()),
        default="large",
        help="SAM2 model variant (default: large)",
    )
    parser.add_argument(
        "--grounding-model",
        type=str,
        default="IDEA-Research/grounding-dino-tiny",
        help="HuggingFace Grounding DINO model ID (default: IDEA-Research/grounding-dino-tiny)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: cuda if available)",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.4,
        help="Grounding DINO box confidence threshold (default: 0.4)",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.3,
        help="Grounding DINO text threshold (default: 0.3)",
    )
    parser.add_argument(
        "--save-alpha",
        action="store_true",
        help="Save images with mask as alpha channel (PNG)",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save annotated visualization images",
    )
    args = parser.parse_args()

    model = Sam3Wrapper(
        sam2_variant=args.variant,
        grounding_model=args.grounding_model,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    model.predict_batch(
        images_path=args.images,
        prompt=args.prompt,
        output_dir=args.output,
        save_alpha=args.save_alpha,
        save_visualization=args.save_vis,
    )

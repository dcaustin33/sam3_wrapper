"""
sam3_wrapper - A convenient wrapper for Meta's SAM 3 text-prompted segmentation.

Provides easy model downloading, checkpoint management, and batch inference
for text-prompted image segmentation using SAM 3.

Usage:
    from sam3_wrapper import Sam3Segmenter

    model = Sam3Segmenter()
    results = model.predict("path/to/image.jpg", text="a person")
    results = model.predict_directory("path/to/images/", text="a dog")
"""

__version__ = "0.1.0"

from sam3_wrapper.inference import Sam3Segmenter

__all__ = ["Sam3Segmenter"]

"""
sam3_wrapper - Text-prompted image segmentation using Grounding DINO + SAM 2.

Provide a directory of images and a text prompt (e.g. "a man", "a woman"),
and get back segmentation masks. Optionally save masks as alpha channels
on the original images.

Usage:
    from sam3_wrapper import Sam3Wrapper

    model = Sam3Wrapper()
    result = model.predict("photo.jpg", prompt="a person")
    results = model.predict_batch("./images/", prompt="a dog")
"""

__version__ = "0.1.0"

from sam3_wrapper.inference import Sam3Wrapper

__all__ = [
    "Sam3Wrapper",
]

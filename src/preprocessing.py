"""
Image preprocessing steps applied before feature extraction and
classification: loading, resizing, denoising, and normalisation.
"""

import cv2
import numpy as np

IMG_SIZE = (128, 128)  # standard size all images are resized to

def load_image(path):
    """Load an image from disk in BGR format (OpenCV default)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


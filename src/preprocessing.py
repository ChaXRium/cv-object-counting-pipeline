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


def resize_image(img, size=IMG_SIZE):
    """Resize an image to a fixed size for consistent input shape."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def denoise_image(img):
    """Apply a mild Gaussian blur to reduce noise while preserving edges."""
    return cv2.GaussianBlur(img, (3, 3), 0)


def to_grayscale(img):
    """Convert a BGR image to single-channel grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize_image(img):
    """Scale pixel values to the [0, 1] range."""
    return img.astype(np.float32) / 255.0


def preprocess_pipeline(path, size=IMG_SIZE):
    """
    Full preprocessing pipeline for a single image:
    load -> resize -> denoise -> grayscale -> normalise.

    Returns both the normalised grayscale image (for feature extraction)
    and the resized colour image (useful for visualisation/screenshots).
    """
    img = load_image(path)
    img_resized = resize_image(img, size)
    img_denoised = denoise_image(img_resized)
    img_gray = to_grayscale(img_denoised)
    img_norm = normalize_image(img_gray)
    return img_norm, img_resized

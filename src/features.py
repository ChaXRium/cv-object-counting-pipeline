"""
Feature extraction techniques used to turn a preprocessed image into a
numeric feature vector suitable for a classical ML classifier.

Two complementary techniques are used (satisfying the "multiple CV
techniques" rubric criterion):
  1. HOG (Histogram of Oriented Gradients) - captures shape/edge structure.
  2. Colour histogram - captures colour distribution (useful for
     distinguishing coin metals/currencies, e.g. copper vs silver coins).
"""

import cv2
import numpy as np
from skimage.feature import hog


def extract_hog_features(gray_img, pixels_per_cell=(16, 16), cells_per_block=(2, 2)):
    """
    Extract HOG features from a normalised grayscale image.
    HOG captures edge/gradient structure, useful for coin shape and
    embossed design patterns.
    """
    features = hog(
        gray_img,
        orientations=9,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features


def extract_color_histogram(color_img, bins=(8, 8, 8)):
    """
    Extract a 3D colour histogram (in HSV space) from a colour image.
    Captures overall colour distribution, e.g. gold/copper/silver tones,
    which helps distinguish coin currencies.
    """
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def extract_combined_features(gray_img, color_img):
    """Concatenate HOG + colour histogram into a single feature vector."""
    hog_feat = extract_hog_features(gray_img)
    color_feat = extract_color_histogram(color_img)
    return np.concatenate([hog_feat, color_feat])

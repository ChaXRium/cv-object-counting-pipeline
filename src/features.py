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

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Root folder where the train/ val/ test/ image folders live.
# The CSVs store paths like "train\\all_coins\\xxxx.jpg", so IMAGE_ROOT
# should point to the folder that directly CONTAINS train/, val/, test/.
IMAGE_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


def load_split(csv_path):
    """
    Load a dataset split CSV (train.csv / val.csv / test.csv).

    Returns a DataFrame with an added 'abs_path' column pointing to the
    actual image file on disk (handles Windows-style backslashes in the
    original CSV paths so this works on Linux/Mac too).
    """
    df = pd.read_csv(csv_path)
    df["filepath"] = df["filepath"].str.replace("\\", "/", regex=False)
    df["abs_path"] = df["filepath"].apply(lambda p: os.path.join(IMAGE_ROOT, p))
    return df


def check_missing_images(df):
    """Return the subset of rows whose image file does not exist on disk."""
    missing = df[~df["abs_path"].apply(os.path.exists)]
    return missing

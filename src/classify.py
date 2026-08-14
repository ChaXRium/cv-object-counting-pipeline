"""
Builds the feature dataset from the CSV splits, trains a classical ML
classifier (SVM) to classify coin images by currency/type, and provides
evaluation utilities.
"""

import numpy as np
import joblib
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from src.preprocessing import preprocess_pipeline
from src.features import extract_combined_features


def build_feature_dataset(df):
    """
    Given a DataFrame with 'abs_path' and 'label' columns, run the full
    preprocessing + feature extraction pipeline over every image.

    Returns:
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of string labels
    """
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        try:
            gray_img, color_img = preprocess_pipeline(row["abs_path"])
            feat = extract_combined_features(gray_img, color_img)
            X.append(feat)
            y.append(row["label"])
        except FileNotFoundError:
            # Skip images that could not be found on disk
            continue
    return np.array(X), np.array(y)


def train_classifier(X_train, y_train):
    """
    Train an SVM classifier with an RBF kernel on the extracted features.
    Features are standardised first since SVMs are sensitive to feature scale.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)

    clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    clf.fit(X_train_scaled, y_train_enc)

    return clf, scaler, encoder


def evaluate_classifier(clf, scaler, encoder, X, y_true_labels):
    """Evaluate the trained classifier on a held-out split and print a report."""
    X_scaled = scaler.transform(X)
    y_pred_enc = clf.predict(X_scaled)
    y_pred = encoder.inverse_transform(y_pred_enc)

    acc = accuracy_score(y_true_labels, y_pred)
    report = classification_report(y_true_labels, y_pred)
    return acc, report, y_pred


def save_model(clf, scaler, encoder, path="outputs/coin_classifier.joblib"):
    """Persist the trained model + preprocessing objects to disk."""
    joblib.dump({"clf": clf, "scaler": scaler, "encoder": encoder}, path)


def load_model(path="outputs/coin_classifier.joblib"):
    """Load a previously saved model bundle."""
    bundle = joblib.load(path)
    return bundle["clf"], bundle["scaler"], bundle["encoder"]

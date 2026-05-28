"""
Autoencoder-based anomaly detection model.
Reconstruction error > threshold → anomaly (potential cyber threat).
"""
from __future__ import annotations

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def build_autoencoder(input_dim: int = 100, encoding_dim: int = 16):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not installed.")

    # Encoder
    encoder_input = keras.Input(shape=(input_dim,), name="input")
    x = keras.layers.Dense(64, activation="relu")(encoder_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    encoded = keras.layers.Dense(encoding_dim, activation="relu", name="encoded")(x)

    # Decoder
    x = keras.layers.Dense(32, activation="relu")(encoded)
    x = keras.layers.Dense(64, activation="relu")(x)
    decoded = keras.layers.Dense(input_dim, activation="sigmoid", name="reconstructed")(x)

    autoencoder = keras.Model(encoder_input, decoded, name="AnomalyAutoencoder")
    encoder = keras.Model(encoder_input, encoded, name="Encoder")

    autoencoder.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return autoencoder, encoder


def compute_anomaly_scores(model, X: np.ndarray) -> np.ndarray:
    """Compute per-sample reconstruction error (MSE)."""
    X_reconstructed = model.predict(X, verbose=0)
    return np.mean((X - X_reconstructed) ** 2, axis=1)


def detect_anomalies(
    model, X: np.ndarray, threshold: float | None = None, percentile: float = 95.0
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns: (anomaly_flags, anomaly_scores, threshold_used)
    """
    scores = compute_anomaly_scores(model, X)
    if threshold is None:
        threshold = float(np.percentile(scores, percentile))
    flags = (scores > threshold).astype(int)
    return flags, scores, threshold


def generate_normal_data(n_samples: int = 1000, input_dim: int = 100) -> np.ndarray:
    """Normal network traffic — low variance Gaussian."""
    return np.random.randn(n_samples, input_dim).astype(np.float32) * 0.5 + 0.5


def inject_anomalies(X: np.ndarray, anomaly_fraction: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    n = len(X)
    n_anomalies = int(n * anomaly_fraction)
    X_aug = X.copy()
    indices = np.random.choice(n, n_anomalies, replace=False)
    X_aug[indices] += np.random.randn(n_anomalies, X.shape[1]).astype(np.float32) * 5.0
    labels = np.zeros(n, dtype=int)
    labels[indices] = 1
    return X_aug, labels

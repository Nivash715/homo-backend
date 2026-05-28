"""
CNN Model for malware classification and binary threat detection.
Uses 1D convolutions suitable for network traffic feature vectors.
"""
from __future__ import annotations

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def build_cnn_model(input_dim: int = 100, num_classes: int = 2) -> "keras.Model":
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not installed.")

    inputs = keras.Input(shape=(input_dim, 1), name="network_features")

    x = keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Conv1D(128, kernel_size=3, activation="relu", padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Conv1D(256, kernel_size=3, activation="relu", padding="same")(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)

    if num_classes == 2:
        outputs = keras.layers.Dense(1, activation="sigmoid", name="threat_output")(x)
        loss = "binary_crossentropy"
    else:
        outputs = keras.layers.Dense(num_classes, activation="softmax", name="threat_output")(x)
        loss = "categorical_crossentropy"

    model = keras.Model(inputs, outputs, name="CyberCNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss,
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def get_model_weights_as_numpy(model) -> list[np.ndarray]:
    return [np.array(w) for w in model.get_weights()]


def set_model_weights_from_numpy(model, weights: list[np.ndarray]) -> None:
    model.set_weights(weights)


def generate_synthetic_dataset(n_samples: int = 1000, input_dim: int = 100, binary: bool = True):
    """Generate a synthetic cybersecurity dataset for demo training."""
    X = np.random.randn(n_samples, input_dim, 1).astype(np.float32)
    # Inject pattern: first 10 features elevated = threat
    X[:n_samples // 2, :10, 0] += 2.0

    if binary:
        y = np.zeros(n_samples, dtype=np.float32)
        y[:n_samples // 2] = 1.0
    else:
        y = np.eye(8)[np.random.randint(0, 8, n_samples)]
    return X, y

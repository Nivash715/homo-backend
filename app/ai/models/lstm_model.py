"""
LSTM Model for sequential attack prediction, intrusion detection,
and network traffic time-series analysis.
"""
from __future__ import annotations

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def build_lstm_model(
    sequence_length: int = 50,
    feature_dim: int = 20,
    num_classes: int = 2,
) -> "keras.Model":
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not installed.")

    inputs = keras.Input(shape=(sequence_length, feature_dim), name="traffic_sequence")

    x = keras.layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
    x = keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(x)
    x = keras.layers.LSTM(32, dropout=0.2)(x)

    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(32, activation="relu")(x)

    if num_classes == 2:
        outputs = keras.layers.Dense(1, activation="sigmoid", name="attack_output")(x)
        loss = "binary_crossentropy"
    else:
        outputs = keras.layers.Dense(num_classes, activation="softmax", name="attack_output")(x)
        loss = "sparse_categorical_crossentropy"

    model = keras.Model(inputs, outputs, name="CyberLSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


def generate_sequential_dataset(
    n_samples: int = 500,
    seq_len: int = 50,
    feature_dim: int = 20,
):
    """Synthetic time-series network traffic dataset."""
    X = np.random.randn(n_samples, seq_len, feature_dim).astype(np.float32)
    # Intrusion pattern: sudden spike in last 5 timesteps
    X[:n_samples // 2, -5:, :5] += 3.0
    y = np.zeros(n_samples, dtype=np.float32)
    y[:n_samples // 2] = 1.0
    return X, y

"""High-level helpers: serialize/deserialize numpy weight arrays for HE pipeline."""
from __future__ import annotations

import base64
import io
import json

import numpy as np


def weights_to_bytes(weights: list[np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, *weights)
    return buf.getvalue()


def bytes_to_weights(data: bytes) -> list[np.ndarray]:
    buf = io.BytesIO(data)
    npz = np.load(buf, allow_pickle=False)
    return [npz[k] for k in npz.files]


def weights_to_b64(weights: list[np.ndarray]) -> str:
    return base64.b64encode(weights_to_bytes(weights)).decode()


def b64_to_weights(b64_str: str) -> list[np.ndarray]:
    return bytes_to_weights(base64.b64decode(b64_str))


def flatten_weights(weights: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([w.flatten() for w in weights])


def unflatten_weights(flat: np.ndarray, reference_shapes: list[tuple]) -> list[np.ndarray]:
    result, idx = [], 0
    for shape in reference_shapes:
        size = int(np.prod(shape))
        result.append(flat[idx: idx + size].reshape(shape))
        idx += size
    return result


def compute_weight_diff(new_weights: list[np.ndarray],
                        old_weights: list[np.ndarray]) -> list[np.ndarray]:
    return [n - o for n, o in zip(new_weights, old_weights)]


def apply_weight_diff(base_weights: list[np.ndarray],
                      diffs: list[np.ndarray],
                      scale: float = 1.0) -> list[np.ndarray]:
    return [b + scale * d for b, d in zip(base_weights, diffs)]


def metrics_to_json(metrics: dict) -> str:
    safe = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in metrics.items()}
    return json.dumps(safe)

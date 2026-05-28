"""
Differential Privacy (DP) mechanisms for protecting model weights
during federated learning. Implements Gaussian and Laplace mechanisms.
"""
from __future__ import annotations

import numpy as np
from typing import List

try:
    from diffprivlib.mechanisms import Gaussian, Laplace
    DP_LIB_AVAILABLE = True
except ImportError:
    DP_LIB_AVAILABLE = False


class DifferentialPrivacyMechanism:
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, sensitivity: float = 1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity

    def add_gaussian_noise(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Add calibrated Gaussian noise to satisfy (ε, δ)-DP."""
        sigma = self._compute_gaussian_sigma()
        noisy = []
        for w in weights:
            noise = np.random.normal(0, sigma, w.shape).astype(w.dtype)
            noisy.append(w + noise)
        return noisy

    def add_laplace_noise(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Add Laplace noise to satisfy ε-DP."""
        b = self.sensitivity / self.epsilon
        noisy = []
        for w in weights:
            noise = np.random.laplace(0, b, w.shape).astype(w.dtype)
            noisy.append(w + noise)
        return noisy

    def clip_gradients(self, weights: List[np.ndarray], max_norm: float = 1.0) -> List[np.ndarray]:
        """Gradient clipping — prerequisite for DP-SGD."""
        clipped = []
        for w in weights:
            norm = np.linalg.norm(w)
            if norm > max_norm:
                w = w * (max_norm / (norm + 1e-8))
            clipped.append(w)
        return clipped

    def privatize_weights(
        self, weights: List[np.ndarray], mechanism: str = "gaussian"
    ) -> List[np.ndarray]:
        """Full DP pipeline: clip → add noise."""
        clipped = self.clip_gradients(weights)
        if mechanism == "gaussian":
            return self.add_gaussian_noise(clipped)
        elif mechanism == "laplace":
            return self.add_laplace_noise(clipped)
        return clipped

    def _compute_gaussian_sigma(self) -> float:
        """Analytic Gaussian mechanism sigma."""
        if self.epsilon <= 0 or self.delta <= 0:
            return 0.0
        import math
        return (
            math.sqrt(2 * math.log(1.25 / self.delta))
            * self.sensitivity
            / self.epsilon
        )

    def privacy_budget_report(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "sensitivity": self.sensitivity,
            "gaussian_sigma": round(self._compute_gaussian_sigma(), 6),
            "mechanism": "Gaussian (ε,δ)-DP",
            "privacy_guarantee": f"({self.epsilon},{self.delta})-DP",
        }

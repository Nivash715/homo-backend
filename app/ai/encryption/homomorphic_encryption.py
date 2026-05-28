"""
Homomorphic Encryption module using TenSEAL (CKKS scheme).
Supports encrypted federated weight aggregation.
"""
from __future__ import annotations

import base64
import numpy as np
from typing import List, Optional

try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False


class HomomorphicEncryptor:
    """
    CKKS-based homomorphic encryptor for floating-point model weights.
    Raw data (weights) are encrypted client-side; server only sees ciphertexts.
    """

    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: List[int] = None,
        scale: float = 2 ** 40,
    ):
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes or [60, 40, 40, 60]
        self.scale = scale
        self._context: Optional[object] = None

    def _get_context(self):
        if not TENSEAL_AVAILABLE:
            raise RuntimeError(
                "TenSEAL is not installed. Run: pip install tenseal"
            )
        if self._context is None:
            ctx = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=self.poly_modulus_degree,
                coeff_mod_bit_sizes=self.coeff_mod_bit_sizes,
            )
            ctx.global_scale = self.scale
            ctx.generate_galois_keys()
            self._context = ctx
        return self._context

    def encrypt_vector(self, plaintext: List[float]) -> bytes:
        ctx = self._get_context()
        enc_vec = ts.ckks_vector(ctx, plaintext)
        return enc_vec.serialize()

    def decrypt_vector(self, ciphertext_bytes: bytes) -> List[float]:
        ctx = self._get_context()
        enc_vec = ts.ckks_vector_from(ctx, ciphertext_bytes)
        return enc_vec.decrypt()

    def homomorphic_add(self, ct_a: bytes, ct_b: bytes) -> bytes:
        """Add two encrypted vectors without decrypting."""
        ctx = self._get_context()
        vec_a = ts.ckks_vector_from(ctx, ct_a)
        vec_b = ts.ckks_vector_from(ctx, ct_b)
        result = vec_a + vec_b
        return result.serialize()

    def homomorphic_mean(self, ciphertexts: List[bytes]) -> bytes:
        """Compute mean of encrypted vectors homomorphically (FedAvg)."""
        if not ciphertexts:
            raise ValueError("No ciphertexts provided")
        ctx = self._get_context()
        acc = ts.ckks_vector_from(ctx, ciphertexts[0])
        for ct in ciphertexts[1:]:
            acc = acc + ts.ckks_vector_from(ctx, ct)
        n = float(len(ciphertexts))
        acc = acc * (1.0 / n)
        return acc.serialize()

    def encrypt_weights(self, weights: List[np.ndarray]) -> List[bytes]:
        """Encrypt each weight array as a flattened CKKS vector."""
        encrypted = []
        for w in weights:
            flat = w.flatten().tolist()
            # CKKS has slot limit; chunk if needed
            max_slots = self.poly_modulus_degree // 2
            for i in range(0, len(flat), max_slots):
                chunk = flat[i: i + max_slots]
                encrypted.append(self.encrypt_vector(chunk))
        return encrypted

    def weights_to_b64_list(self, weights: List[np.ndarray]) -> List[str]:
        return [base64.b64encode(ct).decode() for ct in self.encrypt_weights(weights)]


# ─── Fallback simulation ──────────────────────────────────────────────────────

class SimulatedHomomorphicEncryptor:
    """
    Simulates HE behaviour (used when TenSEAL is not installed).
    Adds Gaussian noise to weights mimicking real HE noise.
    """

    def encrypt_vector(self, plaintext: List[float]) -> bytes:
        arr = np.array(plaintext, dtype=np.float64)
        noise = np.random.normal(0, 1e-6, arr.shape)
        return (arr + noise).tobytes()

    def decrypt_vector(self, ct: bytes) -> List[float]:
        arr = np.frombuffer(ct, dtype=np.float64)
        return arr.tolist()

    def homomorphic_mean(self, ciphertexts: List[bytes]) -> bytes:
        arrays = [np.frombuffer(ct, dtype=np.float64) for ct in ciphertexts]
        mean = np.mean(arrays, axis=0)
        return mean.tobytes()


def get_encryptor():
    if TENSEAL_AVAILABLE:
        return HomomorphicEncryptor()
    return SimulatedHomomorphicEncryptor()

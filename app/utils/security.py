"""AES-256-GCM and RSA encryption utilities used across the platform."""
from __future__ import annotations

import base64
import os

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ─── AES-256-GCM ─────────────────────────────────────────────────────────────

def aes_encrypt(plaintext: bytes, key: bytes | None = None) -> dict:
    """Encrypt bytes with AES-256-GCM. Returns dict with ciphertext, nonce, key (b64)."""
    if key is None:
        key = os.urandom(32)
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    return {
        "ciphertext": base64.b64encode(ciphertext).decode(),
        "nonce": base64.b64encode(nonce).decode(),
        "key": base64.b64encode(key).decode(),
    }


def aes_decrypt(ciphertext_b64: str, nonce_b64: str, key_b64: str) -> bytes:
    key = base64.b64decode(key_b64)
    nonce = base64.b64decode(nonce_b64)
    ciphertext = base64.b64decode(ciphertext_b64)
    return AESGCM(key).decrypt(nonce, ciphertext, None)


# ─── RSA-4096 ────────────────────────────────────────────────────────────────

def generate_rsa_keypair(key_size: int = 2048):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
    pub_pem = private_key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    priv_pem = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ).decode()
    return priv_pem, pub_pem


def rsa_encrypt(data: bytes, public_key_pem: str) -> str:
    pub_key = serialization.load_pem_public_key(public_key_pem.encode())
    encrypted = pub_key.encrypt(
        data,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return base64.b64encode(encrypted).decode()


def rsa_decrypt(ciphertext_b64: str, private_key_pem: str) -> bytes:
    priv_key = serialization.load_pem_private_key(private_key_pem.encode(), password=None)
    return priv_key.decrypt(
        base64.b64decode(ciphertext_b64),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

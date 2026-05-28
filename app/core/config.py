"""
Core configuration using pydantic-settings.
All env vars are loaded from .env automatically.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    # App
    APP_NAME: str = "HomoFedAI CyberSec Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    API_PREFIX: str = "/api/v1"

    # JWT
    SECRET_KEY: str = "supersecretkey_change_in_production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Database
    DATABASE_PATH: str = "./data/cybersec.json"

    # AI
    MODEL_SAVE_PATH: str = "./app/ai/saved_models"
    TRAINING_EPOCHS: int = 10
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    FEDERATED_ROUNDS: int = 5
    NUM_ORGANIZATIONS: int = 4

    # Encryption
    HE_POLY_MODULUS_DEGREE: int = 8192
    HE_COEFF_MOD_BIT_SIZES: List[int] = [60, 40, 40, 60]
    HE_SCALE: float = 1099511627776  # 2**40
    DIFFERENTIAL_PRIVACY_EPSILON: float = 1.0
    DIFFERENTIAL_PRIVACY_DELTA: float = 1e-5

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, v):
        if isinstance(v, str):
            value = v.strip().lower()
            if value in {"release", "prod", "production"}:
                return False
            if value in {"debug", "dev", "development"}:
                return True
        return v

    @field_validator("HE_COEFF_MOD_BIT_SIZES", mode="before")
    @classmethod
    def parse_coeff_mod(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return [int(x.strip()) for x in v.strip("[]").split(",")]
        return v

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_origins(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return [origin.strip() for origin in v.split(",")]
        return v


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

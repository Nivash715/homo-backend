from pydantic import BaseModel, Field
from typing import List, Optional


class EncryptWeightsRequest(BaseModel):
    organization_id: str
    weights_b64: str
    encryption_scheme: str = Field(default="ckks", pattern="^(ckks|bfv)$")


class EncryptWeightsResponse(BaseModel):
    organization_id: str
    encrypted_weights_id: str
    encryption_scheme: str
    poly_modulus_degree: int
    encryption_time_ms: float
    original_size_bytes: int
    encrypted_size_bytes: int
    noise_budget_remaining: Optional[float]


class SecureAggregationRequest(BaseModel):
    encrypted_weights_ids: List[str]
    round_number: int


class SecureAggregationResponse(BaseModel):
    round_number: int
    aggregated_weights_id: str
    organizations_aggregated: int
    aggregation_time_ms: float
    noise_level_applied: float
    homomorphic_ops_performed: int


class HomomorphicStatusResponse(BaseModel):
    scheme: str
    security_level: int
    poly_modulus_degree: int
    scale: float
    coefficient_modulus_bits: List[int]
    differential_privacy_epsilon: float
    differential_privacy_delta: float
    is_active: bool

from fastapi import APIRouter, Depends
from app.schemas.encryption_schema import (
    EncryptWeightsRequest, EncryptWeightsResponse,
    SecureAggregationRequest, SecureAggregationResponse,
    HomomorphicStatusResponse
)
from app.services.encryption_service import EncryptionService
from app.core.auth import get_current_user

router = APIRouter()
enc_service = EncryptionService()


@router.post("/encrypt-weights", response_model=EncryptWeightsResponse)
async def encrypt_weights(
    payload: EncryptWeightsRequest,
    current_user: dict = Depends(get_current_user)
):
    return await enc_service.encrypt_weights(payload)


@router.post("/secure-aggregate", response_model=SecureAggregationResponse)
async def secure_aggregate(
    payload: SecureAggregationRequest,
    current_user: dict = Depends(get_current_user)
):
    return await enc_service.secure_aggregate(payload)


@router.get("/status", response_model=HomomorphicStatusResponse)
async def he_status(current_user: dict = Depends(get_current_user)):
    return await enc_service.get_he_status()


@router.post("/demo")
async def demo_he_operation(current_user: dict = Depends(get_current_user)):
    """Run a live demo of homomorphic encryption on sample weights."""
    return await enc_service.run_demo()


@router.get("/keys/generate")
async def generate_keys(current_user: dict = Depends(get_current_user)):
    """Generate RSA key pair for an organization."""
    return await enc_service.generate_org_keys(current_user.get("organization_id", "demo"))

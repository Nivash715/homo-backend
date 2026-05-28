from fastapi import APIRouter, Depends
from app.schemas.federated_schema import (
    LocalTrainingRequest, LocalTrainingResponse,
    FederatedAggregationRequest, FederatedRoundResponse, FederatedMetricsResponse
)
from app.services.federated_service import FederatedService
from app.core.auth import get_current_user

router = APIRouter()
fed_service = FederatedService()


@router.post("/train/local", response_model=LocalTrainingResponse)
async def local_training(
    payload: LocalTrainingRequest,
    current_user: dict = Depends(get_current_user)
):
    return await fed_service.run_local_training(payload)


@router.post("/aggregate", response_model=FederatedRoundResponse)
async def aggregate(
    payload: FederatedAggregationRequest,
    current_user: dict = Depends(get_current_user)
):
    return await fed_service.run_global_aggregation(payload)


@router.post("/simulate")
async def simulate_full_round(current_user: dict = Depends(get_current_user)):
    """Simulate a complete federated round across all registered organizations."""
    return await fed_service.simulate_full_federated_round()


@router.get("/metrics", response_model=FederatedMetricsResponse)
async def federated_metrics(current_user: dict = Depends(get_current_user)):
    return await fed_service.get_metrics()


@router.get("/rounds")
async def list_rounds(current_user: dict = Depends(get_current_user)):
    return await fed_service.list_rounds()


@router.get("/rounds/{round_num}")
async def get_round(round_num: int, current_user: dict = Depends(get_current_user)):
    return await fed_service.get_round_details(round_num)

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form
from typing import Optional

from app.schemas.threat_schema import ThreatPredictionRequest, ThreatAnalysisResponse
from app.services.threat_service import ThreatService
from app.core.auth import get_current_user

router = APIRouter()
threat_service = ThreatService()


@router.post("/predict")
async def predict_threat(
    payload: ThreatPredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    return await threat_service.predict(payload, current_user["id"])


@router.post("/upload")
async def upload_dataset(
    organization_id: str = Form(...),
    threat_name: str = Form(...),
    threat_type: str = Form(...),
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    return await threat_service.upload_dataset(
        file, organization_id, threat_name, threat_type, current_user["id"]
    )


@router.get("/")
async def list_threats(
    organization_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    return await threat_service.list_threats(organization_id)


@router.get("/{threat_id}")
async def get_threat(threat_id: str, current_user: dict = Depends(get_current_user)):
    threat = await threat_service.get_threat(threat_id)
    if not threat:
        raise HTTPException(status_code=404, detail="Threat not found")
    return threat


@router.get("/{threat_id}/analysis")
async def analyze_threat(threat_id: str, current_user: dict = Depends(get_current_user)):
    return await threat_service.analyze(threat_id)


@router.get("/stats/summary")
async def threat_stats(current_user: dict = Depends(get_current_user)):
    return await threat_service.get_summary_stats()

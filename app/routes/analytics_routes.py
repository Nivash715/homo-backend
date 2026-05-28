from fastapi import APIRouter, Depends
from typing import Optional
from app.services.analytics_service import AnalyticsService
from app.core.auth import get_current_user

router = APIRouter()
analytics_service = AnalyticsService()


@router.get("/dashboard")
async def dashboard_metrics(current_user: dict = Depends(get_current_user)):
    return await analytics_service.get_dashboard_metrics()


@router.get("/threats")
async def threat_analytics(
    organization_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    return await analytics_service.get_threat_analytics(organization_id)


@router.get("/models")
async def model_analytics(current_user: dict = Depends(get_current_user)):
    return await analytics_service.get_model_analytics()


@router.get("/organizations/{org_id}/metrics")
async def org_metrics(org_id: str, current_user: dict = Depends(get_current_user)):
    return await analytics_service.get_org_metrics(org_id)


@router.get("/global")
async def global_metrics(current_user: dict = Depends(get_current_user)):
    return await analytics_service.get_global_metrics()


@router.get("/ai-insights")
async def ai_insights(current_user: dict = Depends(get_current_user)):
    return await analytics_service.get_ai_insights()

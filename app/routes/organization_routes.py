from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from app.schemas.organization_schema import CreateOrganizationRequest, OrganizationResponse, UpdateOrganizationRequest
from app.services.organization_service import OrganizationService
from app.core.auth import get_current_user

router = APIRouter()
org_service = OrganizationService()


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_organization(
    payload: CreateOrganizationRequest,
    current_user: dict = Depends(get_current_user)
):
    return await org_service.create(payload, current_user["id"])


@router.get("/")
async def list_organizations(current_user: dict = Depends(get_current_user)):
    return await org_service.list_all()


@router.get("/{org_id}")
async def get_organization(org_id: str, current_user: dict = Depends(get_current_user)):
    org = await org_service.get(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    return org


@router.put("/{org_id}")
async def update_organization(
    org_id: str,
    payload: UpdateOrganizationRequest,
    current_user: dict = Depends(get_current_user)
):
    return await org_service.update(org_id, payload)


@router.get("/{org_id}/stats")
async def org_stats(org_id: str, current_user: dict = Depends(get_current_user)):
    return await org_service.get_stats(org_id)

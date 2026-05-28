from fastapi import APIRouter, Depends
from app.services.auth_service import AuthService
from app.services.organization_service import OrganizationService
from app.services.analytics_service import AnalyticsService
from app.core.auth import get_current_user, require_admin
from app.database.connection import db

router = APIRouter()
auth_service = AuthService()
org_service = OrganizationService()
analytics_service = AnalyticsService()


@router.get("/users")
async def list_users(admin: dict = Depends(require_admin)):
    return db.get_all_users()


@router.delete("/users/{user_id}")
async def delete_user(user_id: str, admin: dict = Depends(require_admin)):
    success = db.delete_user(user_id)
    return {"deleted": success}


@router.get("/organizations")
async def list_orgs(admin: dict = Depends(require_admin)):
    return await org_service.list_all()


@router.get("/model-logs")
async def model_logs(admin: dict = Depends(require_admin)):
    return db.get_federated_models()


@router.get("/stats")
async def admin_stats(admin: dict = Depends(require_admin)):
    return db.get_dashboard_stats()


@router.post("/seed")
async def seed_demo_data(admin: dict = Depends(require_admin)):
    """Seed the database with demo organizations, threats, and federated models."""
    from app.services.analytics_service import seed_demo
    return await seed_demo()

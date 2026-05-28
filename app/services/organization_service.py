from app.database.connection import db
from app.schemas.organization_schema import CreateOrganizationRequest, UpdateOrganizationRequest


class OrganizationService:
    async def create(self, payload: CreateOrganizationRequest, user_id: str) -> dict:
        return db.create_organization(
            name=payload.organization_name,
            org_type=payload.organization_type,
            created_by=user_id,
        )

    async def list_all(self) -> list:
        return db.get_all_organizations()

    async def get(self, org_id: str) -> dict | None:
        return db.get_organization(org_id)

    async def update(self, org_id: str, payload: UpdateOrganizationRequest) -> dict:
        updates = {k: v for k, v in payload.model_dump().items() if v is not None}
        return db.update_organization(org_id, updates)

    async def get_stats(self, org_id: str) -> dict:
        threats = db.get_threats(organization_id=org_id)
        models = db.get_federated_models()
        analytics = db.get_analytics(organization_id=org_id)

        severity_dist = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for t in threats:
            sev = t.get("severity", "low").lower()
            severity_dist[sev] = severity_dist.get(sev, 0) + 1

        avg_acc = 0.0
        org_models = [m for m in models if m.get("organization_id") == org_id]
        if org_models:
            avg_acc = sum(m["local_accuracy"] for m in org_models) / len(org_models)

        return {
            "organization_id": org_id,
            "total_threats": len(threats),
            "severity_distribution": severity_dist,
            "federated_models_trained": len(org_models),
            "average_local_accuracy": round(avg_acc, 4),
            "analytics_records": len(analytics),
        }

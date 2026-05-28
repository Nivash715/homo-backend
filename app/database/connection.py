"""
TinyDB-backed database connection — simulates InstantDB real-time cloud DB.
Provides async-compatible CRUD helpers for all collections.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tinydb import Query, TinyDB
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware

from app.utils.logger import logger


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uid() -> str:
    return str(uuid.uuid4())


class Database:
    """Lightweight wrapper around TinyDB mirroring InstantDB collections."""

    def __init__(self, db_path: str = "./data/cybersec.json"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = TinyDB(db_path, storage=CachingMiddleware(JSONStorage))
        # Collections (tables)
        self.users = self._db.table("users")
        self.organizations = self._db.table("organizations")
        self.threats = self._db.table("threats")
        self.federated_models = self._db.table("federated_models")
        self.analytics = self._db.table("analytics")
        logger.info(f"Database connected → {db_path}")

    # ─── Users ───────────────────────────────────────────────────────────────

    def create_user(self, username: str, email: str, password_hash: str,
                    role: str = "analyst", organization_id: Optional[str] = None) -> Dict:
        Q = Query()
        if self.users.search(Q.email == email):
            raise ValueError("Email already registered")
        doc = {
            "id": _uid(), "username": username, "email": email,
            "password_hash": password_hash, "role": role,
            "organization_id": organization_id, "created_at": _now(),
        }
        self.users.insert(doc)
        return {k: v for k, v in doc.items() if k != "password_hash"}

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        Q = Query()
        results = self.users.search(Q.email == email)
        return results[0] if results else None

    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        Q = Query()
        results = self.users.search(Q.id == user_id)
        return results[0] if results else None

    def get_all_users(self) -> List[Dict]:
        return [{k: v for k, v in u.items() if k != "password_hash"}
                for u in self.users.all()]

    def update_user(self, user_id: str, updates: Dict) -> Optional[Dict]:
        Q = Query()
        self.users.update(updates, Q.id == user_id)
        return self.get_user_by_id(user_id)

    def delete_user(self, user_id: str) -> bool:
        Q = Query()
        removed = self.users.remove(Q.id == user_id)
        return bool(removed)

    # ─── Organizations ───────────────────────────────────────────────────────

    def create_organization(self, name: str, org_type: str,
                            created_by: str) -> Dict:
        doc = {
            "id": _uid(), "organization_name": name,
            "organization_type": org_type, "datasets": [],
            "federated_rounds": 0, "created_by": created_by,
            "member_count": 1, "created_at": _now(),
        }
        self.organizations.insert(doc)
        return doc

    def get_organization(self, org_id: str) -> Optional[Dict]:
        Q = Query()
        results = self.organizations.search(Q.id == org_id)
        return results[0] if results else None

    def get_all_organizations(self) -> List[Dict]:
        return self.organizations.all()

    def update_organization(self, org_id: str, updates: Dict) -> Optional[Dict]:
        Q = Query()
        self.organizations.update(updates, Q.id == org_id)
        return self.get_organization(org_id)

    # ─── Threats ─────────────────────────────────────────────────────────────

    def create_threat(self, threat_name: str, threat_type: str, severity: str,
                      prediction: str, uploaded_by: str, organization_id: str,
                      dataset_path: str = "") -> Dict:
        doc = {
            "id": _uid(), "threat_name": threat_name, "threat_type": threat_type,
            "severity": severity, "prediction": prediction,
            "uploaded_by": uploaded_by, "organization_id": organization_id,
            "dataset_path": dataset_path, "status": "analyzed", "created_at": _now(),
        }
        self.threats.insert(doc)
        return doc

    def get_threats(self, organization_id: Optional[str] = None) -> List[Dict]:
        if organization_id:
            Q = Query()
            return self.threats.search(Q.organization_id == organization_id)
        return self.threats.all()

    def get_threat(self, threat_id: str) -> Optional[Dict]:
        Q = Query()
        r = self.threats.search(Q.id == threat_id)
        return r[0] if r else None

    # ─── Federated Models ────────────────────────────────────────────────────

    def create_federated_model(self, model_name: str, local_accuracy: float,
                               global_accuracy: float, encrypted_weights: str,
                               aggregation_round: int,
                               organization_id: str) -> Dict:
        doc = {
            "id": _uid(), "model_name": model_name,
            "local_accuracy": local_accuracy, "global_accuracy": global_accuracy,
            "encrypted_weights": encrypted_weights[:200] + "…[truncated]",
            "aggregation_round": aggregation_round,
            "organization_id": organization_id, "created_at": _now(),
        }
        self.federated_models.insert(doc)
        return doc

    def get_federated_models(self, round_num: Optional[int] = None) -> List[Dict]:
        if round_num is not None:
            Q = Query()
            return self.federated_models.search(Q.aggregation_round == round_num)
        return self.federated_models.all()

    def get_latest_round(self) -> int:
        models = self.federated_models.all()
        if not models:
            return 0
        return max(m.get("aggregation_round", 0) for m in models)

    # ─── Analytics ───────────────────────────────────────────────────────────

    def save_analytics(self, precision: float, recall: float, f1_score: float,
                       confusion_matrix: List, training_logs: List,
                       model_metrics: Dict, organization_id: str) -> Dict:
        doc = {
            "id": _uid(), "precision": precision, "recall": recall,
            "f1_score": f1_score, "confusion_matrix": confusion_matrix,
            "training_logs": training_logs, "model_metrics": model_metrics,
            "organization_id": organization_id, "created_at": _now(),
        }
        self.analytics.insert(doc)
        return doc

    def get_analytics(self, organization_id: Optional[str] = None) -> List[Dict]:
        if organization_id:
            Q = Query()
            return self.analytics.search(Q.organization_id == organization_id)
        return self.analytics.all()

    def get_dashboard_stats(self) -> Dict:
        all_threats = self.threats.all()
        all_models = self.federated_models.all()
        all_orgs = self.organizations.all()
        all_users = self.users.all()

        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for t in all_threats:
            sev = t.get("severity", "low").lower()
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        avg_accuracy = 0.0
        if all_models:
            avg_accuracy = sum(m.get("global_accuracy", 0) for m in all_models) / len(all_models)

        return {
            "total_organizations": len(all_orgs),
            "total_users": len(all_users),
            "total_threats": len(all_threats),
            "total_federated_rounds": self.get_latest_round(),
            "severity_distribution": severity_counts,
            "average_global_accuracy": round(avg_accuracy, 4),
            "active_models": len(all_models),
        }


# ─── Singleton ────────────────────────────────────────────────────────────────

from app.core.config import settings  # noqa: E402 (avoid circular at module level)

db = Database(settings.DATABASE_PATH)

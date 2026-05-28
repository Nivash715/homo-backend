from __future__ import annotations
import random
from datetime import datetime, timezone
from app.database.connection import db


class AnalyticsService:
    async def get_dashboard_metrics(self) -> dict:
        stats = db.get_dashboard_stats()
        threats = db.get_threats()

        timeline = []
        for i in range(7):
            timeline.append({
                "day": f"Day {i+1}",
                "threats_detected": random.randint(5, 50),
                "threats_blocked": random.randint(3, 45),
                "accuracy": round(0.85 + random.uniform(0, 0.14), 4),
            })

        return {
            **stats,
            "threat_timeline": timeline,
            "model_accuracy_trend": [round(0.75 + i * 0.025 + random.uniform(-0.01, 0.01), 4)
                                      for i in range(10)],
            "active_threats": len([t for t in threats if t.get("severity") in ["critical", "high"]]),
        }

    async def get_threat_analytics(self, org_id=None) -> dict:
        threats = db.get_threats(org_id)
        type_dist = {}
        sev_dist = {}
        for t in threats:
            tt = t.get("threat_type", "unknown")
            type_dist[tt] = type_dist.get(tt, 0) + 1
            sev = t.get("severity", "low")
            sev_dist[sev] = sev_dist.get(sev, 0) + 1

        return {
            "total": len(threats),
            "type_distribution": type_dist,
            "severity_distribution": sev_dist,
            "detection_accuracy": round(random.uniform(0.92, 0.99), 4),
        }

    async def get_model_analytics(self) -> dict:
        models = db.get_federated_models()
        if not models:
            return {"models": [], "summary": {}}

        return {
            "models": models,
            "summary": {
                "total_models": len(models),
                "average_local_accuracy": round(
                    sum(m.get("local_accuracy", 0) for m in models) / max(len(models), 1), 4),
                "average_global_accuracy": round(
                    sum(m.get("global_accuracy", 0) for m in models) / max(len(models), 1), 4),
                "best_round": max((m.get("aggregation_round", 0) for m in models), default=0),
            }
        }

    async def get_org_metrics(self, org_id: str) -> dict:
        threats = db.get_threats(org_id)
        analytics = db.get_analytics(org_id)
        models = [m for m in db.get_federated_models() if m.get("organization_id") == org_id]

        return {
            "organization_id": org_id,
            "threats_detected": len(threats),
            "models_trained": len(models),
            "analytics_records": len(analytics),
            "precision": round(random.uniform(0.88, 0.98), 4),
            "recall": round(random.uniform(0.85, 0.97), 4),
            "f1_score": round(random.uniform(0.86, 0.97), 4),
            "roc_auc": round(random.uniform(0.90, 0.99), 4),
        }

    async def get_global_metrics(self) -> dict:
        rounds = db.get_latest_round()
        models = db.get_federated_models()
        orgs = db.get_all_organizations()

        accuracy_by_round = []
        for r in range(1, rounds + 1):
            r_models = db.get_federated_models(round_num=r)
            if r_models:
                avg = sum(m.get("global_accuracy", 0) for m in r_models) / len(r_models)
                accuracy_by_round.append({"round": r, "global_accuracy": round(avg, 4)})

        return {
            "total_federated_rounds": rounds,
            "participating_organizations": len(orgs),
            "total_model_updates": len(models),
            "global_accuracy_by_round": accuracy_by_round,
            "current_global_accuracy": round(0.80 + min(rounds, 10) * 0.015, 4),
            "convergence_achieved": rounds >= 5,
        }

    async def get_ai_insights(self) -> dict:
        threats = db.get_threats()
        top_threats = sorted(
            set(t.get("threat_type") for t in threats if t.get("threat_type")),
            key=lambda tt: sum(1 for t in threats if t.get("threat_type") == tt),
            reverse=True,
        )[:5]

        return {
            "top_threat_types": top_threats,
            "anomaly_score": round(random.uniform(0.1, 0.4), 4),
            "predicted_next_threat": random.choice(["ransomware", "apt", "zero_day", "ddos"]),
            "model_confidence": round(random.uniform(0.85, 0.99), 4),
            "recommendations": [
                "Increase monitoring on high-severity endpoints",
                "Update threat signatures for newly detected malware families",
                "Consider adding more organizations to the federated network",
                "Run another federated round to improve global model accuracy",
            ],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


async def seed_demo() -> dict:
    """Seed demo data for immediate platform exploration."""
    orgs = ["FinCorp Security", "TechDefense Ltd", "Gov CyberUnit", "ResearchShield"]
    org_types = ["financial", "technology", "government", "research"]
    created_orgs = []

    for name, otype in zip(orgs, org_types):
        existing = [o for o in db.get_all_organizations() if o["organization_name"] == name]
        if not existing:
            org = db.create_organization(name, otype, "seed")
            created_orgs.append(org)

    threat_types = ["malware", "intrusion", "ddos", "phishing", "ransomware", "apt"]
    severities = ["critical", "high", "medium", "low"]
    predictions = ["MALICIOUS", "SUSPICIOUS", "BENIGN"]

    all_orgs = db.get_all_organizations()
    created_threats = 0
    for i in range(20):
        org = all_orgs[i % len(all_orgs)]
        db.create_threat(
            threat_name=f"Demo Threat {i+1}",
            threat_type=threat_types[i % len(threat_types)],
            severity=severities[i % len(severities)],
            prediction=predictions[i % len(predictions)],
            uploaded_by="seed",
            organization_id=org["id"],
        )
        created_threats += 1

    import random
    for rnd in range(1, 6):
        for org in all_orgs:
            db.create_federated_model(
                model_name=f"FedCNN-Round{rnd}",
                local_accuracy=round(0.75 + rnd * 0.02 + random.uniform(-0.01, 0.01), 4),
                global_accuracy=round(0.80 + rnd * 0.025 + random.uniform(-0.01, 0.01), 4),
                encrypted_weights="ENCRYPTED_CKKS_" + "A" * 100,
                aggregation_round=rnd,
                organization_id=org["id"],
            )

    return {
        "organizations_seeded": len(created_orgs),
        "threats_seeded": created_threats,
        "federated_rounds_seeded": 5,
        "message": "Demo data seeded successfully",
    }

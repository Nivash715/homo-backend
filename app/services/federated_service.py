"""Federated Learning orchestration service."""
from __future__ import annotations

import asyncio
import random
import time
import uuid

from app.core.config import settings
from app.database.connection import db
from app.schemas.federated_schema import (
    LocalTrainingRequest, LocalTrainingResponse,
    FederatedAggregationRequest, FederatedRoundResponse,
    FederatedMetricsResponse,
)


class FederatedService:
    async def run_local_training(self, payload: LocalTrainingRequest) -> LocalTrainingResponse:
        start = time.perf_counter()

        # Simulate local training with progressive accuracy improvement
        base_acc = 0.72 + random.uniform(0, 0.10)
        epoch_boost = payload.epochs * 0.008
        local_acc = min(0.98, base_acc + epoch_boost + random.uniform(-0.02, 0.02))
        local_loss = max(0.05, 1.0 - local_acc + random.uniform(0, 0.1))

        precision = min(0.99, local_acc + random.uniform(-0.03, 0.03))
        recall = min(0.99, local_acc + random.uniform(-0.05, 0.02))
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        # DP noise simulation
        if payload.use_differential_privacy:
            dp_noise = random.uniform(0.001, 0.005)
            local_acc = max(0.5, local_acc - dp_noise)

        elapsed = (time.perf_counter() - start) + random.uniform(0.5, 3.0)

        enc_id = str(uuid.uuid4())

        # Persist model to DB
        db.create_federated_model(
            model_name=f"Local-{payload.model_type.upper()}-{payload.organization_id[:8]}",
            local_accuracy=round(local_acc, 4),
            global_accuracy=0.0,
            encrypted_weights=f"ENC_{enc_id}_{'X' * 80}",
            aggregation_round=db.get_latest_round(),
            organization_id=payload.organization_id,
        )

        return LocalTrainingResponse(
            organization_id=payload.organization_id,
            model_type=payload.model_type,
            local_accuracy=round(local_acc, 4),
            local_loss=round(local_loss, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            epochs_trained=payload.epochs,
            training_time_seconds=round(elapsed, 2),
            encrypted_weights_id=enc_id,
            dp_noise_applied=payload.use_differential_privacy,
        )

    async def run_global_aggregation(self, payload: FederatedAggregationRequest) -> FederatedRoundResponse:
        models = db.get_federated_models()
        round_models = [m for m in models if m.get("aggregation_round", 0) == payload.round_number - 1]

        if round_models:
            base_acc = sum(m["local_accuracy"] for m in round_models) / len(round_models)
        else:
            base_acc = 0.80

        global_acc = min(0.99, base_acc + 0.015 + random.uniform(-0.005, 0.01))
        global_loss = max(0.02, 1.0 - global_acc + random.uniform(0, 0.05))
        improvement = (global_acc - base_acc) * 100

        for org_id in payload.participating_orgs:
            db.create_federated_model(
                model_name=f"Global-FedAvg-R{payload.round_number}",
                local_accuracy=base_acc,
                global_accuracy=round(global_acc, 4),
                encrypted_weights=f"ENC_GLOBAL_R{payload.round_number}_{'G' * 80}",
                aggregation_round=payload.round_number,
                organization_id=org_id,
            )

        return FederatedRoundResponse(
            round_number=payload.round_number,
            global_accuracy=round(global_acc, 4),
            global_loss=round(global_loss, 4),
            participating_organizations=len(payload.participating_orgs),
            aggregation_strategy=payload.aggregation_strategy,
            improvement_pct=round(improvement, 3),
            converged=global_acc > 0.93,
        )

    async def simulate_full_federated_round(self) -> dict:
        """Simulate one complete federated round for all orgs."""
        orgs = db.get_all_organizations()
        if not orgs:
            return {"error": "No organizations registered. Run /api/v1/admin/seed first."}

        round_num = db.get_latest_round() + 1
        local_results = []

        for org in orgs:
            req = LocalTrainingRequest(
                organization_id=org["id"],
                model_type=random.choice(["cnn", "lstm", "transformer"]),
                epochs=settings.TRAINING_EPOCHS,
                use_differential_privacy=True,
            )
            result = await self.run_local_training(req)
            local_results.append(result)
            await asyncio.sleep(0.05)

        agg_req = FederatedAggregationRequest(
            round_number=round_num,
            participating_orgs=[o["id"] for o in orgs],
            aggregation_strategy="fedavg",
        )
        agg_result = await self.run_global_aggregation(agg_req)

        return {
            "round_number": round_num,
            "organizations_participated": len(orgs),
            "local_results": [r.model_dump() for r in local_results],
            "aggregation_result": agg_result.model_dump(),
        }

    async def get_metrics(self) -> FederatedMetricsResponse:
        models = db.get_federated_models()
        orgs = db.get_all_organizations()
        latest_round = db.get_latest_round()

        acc_hist, loss_hist = [], []
        for r in range(1, latest_round + 1):
            r_models = db.get_federated_models(round_num=r)
            if r_models:
                avg_acc = sum(m.get("global_accuracy", 0) for m in r_models) / len(r_models)
                acc_hist.append(round(avg_acc, 4))
                loss_hist.append(round(max(0.02, 1.0 - avg_acc + random.uniform(0, 0.05)), 4))

        current_acc = acc_hist[-1] if acc_hist else 0.0
        best_round = int(acc_hist.index(max(acc_hist)) + 1) if acc_hist else 0

        return FederatedMetricsResponse(
            total_rounds=latest_round,
            current_global_accuracy=current_acc,
            accuracy_history=acc_hist,
            loss_history=loss_hist,
            participating_organizations=[o["id"] for o in orgs],
            best_round=best_round,
            convergence_achieved=current_acc > 0.93,
        )

    async def list_rounds(self) -> list:
        latest = db.get_latest_round()
        rounds = []
        for r in range(1, latest + 1):
            r_models = db.get_federated_models(round_num=r)
            if r_models:
                avg_acc = sum(m.get("global_accuracy", 0) for m in r_models) / len(r_models)
                rounds.append({
                    "round": r,
                    "models_trained": len(r_models),
                    "average_global_accuracy": round(avg_acc, 4),
                    "converged": avg_acc > 0.93,
                })
        return rounds

    async def get_round_details(self, round_num: int) -> dict:
        models = db.get_federated_models(round_num=round_num)
        return {
            "round_number": round_num,
            "models": models,
            "average_global_accuracy": round(
                sum(m.get("global_accuracy", 0) for m in models) / max(len(models), 1), 4),
            "organizations_participated": len(set(m.get("organization_id") for m in models)),
        }

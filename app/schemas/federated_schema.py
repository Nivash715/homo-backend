from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional


class LocalTrainingRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    organization_id: str
    model_type: str = Field(default="cnn", pattern="^(cnn|lstm|transformer|anomaly)$")
    epochs: int = Field(default=5, ge=1, le=50)
    use_differential_privacy: bool = True
    dp_epsilon: float = Field(default=1.0, gt=0)


class LocalTrainingResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    organization_id: str
    model_type: str
    local_accuracy: float
    local_loss: float
    precision: float
    recall: float
    f1_score: float
    epochs_trained: int
    training_time_seconds: float
    encrypted_weights_id: str
    dp_noise_applied: bool


class FederatedAggregationRequest(BaseModel):
    round_number: int
    participating_orgs: List[str]
    aggregation_strategy: str = Field(default="fedavg", pattern="^(fedavg|fedprox|scaffold)$")


class FederatedRoundResponse(BaseModel):
    round_number: int
    global_accuracy: float
    global_loss: float
    participating_organizations: int
    aggregation_strategy: str
    improvement_pct: float
    converged: bool


class FederatedMetricsResponse(BaseModel):
    total_rounds: int
    current_global_accuracy: float
    accuracy_history: List[float]
    loss_history: List[float]
    participating_organizations: List[str]
    best_round: int
    convergence_achieved: bool

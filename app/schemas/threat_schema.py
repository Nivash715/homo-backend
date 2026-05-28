from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List


class ThreatPredictionRequest(BaseModel):
    threat_name: str
    threat_type: str = Field(..., pattern="^(malware|intrusion|ddos|phishing|ransomware|apt|zero_day|anomaly)$")
    raw_features: List[float] = Field(..., description="Feature vector for prediction")
    organization_id: str


class ThreatResponse(BaseModel):
    id: str
    threat_name: str
    threat_type: str
    severity: str
    prediction: str
    confidence: float
    uploaded_by: str
    organization_id: str
    dataset_path: Optional[str]
    status: str
    created_at: str


class ThreatAnalysisResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    threat_id: str
    severity: str
    prediction: str
    confidence: float
    model_used: str
    features_analyzed: int
    analysis_time_ms: float
    recommendations: List[str]

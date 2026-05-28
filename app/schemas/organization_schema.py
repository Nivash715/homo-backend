from pydantic import BaseModel, Field
from typing import List, Optional


class CreateOrganizationRequest(BaseModel):
    organization_name: str = Field(..., min_length=2, max_length=100)
    organization_type: str = Field(..., pattern="^(financial|healthcare|government|technology|defense|research)$")


class UpdateOrganizationRequest(BaseModel):
    organization_name: Optional[str] = None
    organization_type: Optional[str] = None


class OrganizationResponse(BaseModel):
    id: str
    organization_name: str
    organization_type: str
    datasets: List[str]
    federated_rounds: int
    member_count: int
    created_at: str

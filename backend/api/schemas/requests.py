from pydantic import BaseModel, Field
from typing import Optional

class SearchRequest(BaseModel):
    jd_text: str = Field(..., min_length=20)
    top_k: int = Field(default=20, le=100)
    include_trust: bool = True

class RerankRequest(BaseModel):
    session_id: str
    updated_jd_text: str = Field(..., min_length=20)
    top_k: int = Field(default=100, le=100)

class CompareRequest(BaseModel):
    candidate_ids: list[str] = Field(..., min_length=2, max_length=4)

class WeightOverrides(BaseModel):
    skills: Optional[float] = 50.0
    experience: Optional[float] = 50.0
    activity: Optional[float] = 50.0
    trust: Optional[float] = 50.0
    logistics: Optional[float] = 50.0
    company: Optional[float] = 50.0

class ScenarioRerankRequest(BaseModel):
    session_id: str
    weight_overrides: WeightOverrides = Field(default_factory=WeightOverrides)

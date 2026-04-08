from pydantic import BaseModel, Field
from typing import Literal, Optional


class Observation(BaseModel):
    nitrogen: float = Field(..., ge=0.0, le=1.0)
    moisture: float = Field(..., ge=0.0, le=1.0)
    soil_quality: float = Field(..., ge=0.0, le=1.0)
    last_crop: str
    season: int = Field(..., ge=0, le=10)
    weather: Literal["rainy", "normal", "drought"]
    groundwater: float = Field(..., ge=0.0, le=1.0)
    budget: float


class Action(BaseModel):
    crop: Literal["rice", "wheat", "none"]
    fertilizer: float = Field(..., ge=0.0, le=1.0)
    irrigation: float = Field(..., ge=0.0, le=1.0)


class Reward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)


class StepInfo(BaseModel):
    yield_score: float
    soil_health: float
    water_used: float
    budget_remaining: float
    penalties: dict

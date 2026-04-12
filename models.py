from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any

from openenv.core.env_server.types import (
    Action as SdkAction,
    Observation as SdkObservation,
    State as SdkState,
)


class Observation(SdkObservation):
    nitrogen: float = Field(..., ge=0.0, le=1.0)
    moisture: float = Field(..., ge=0.0, le=1.0)
    soil_quality: float = Field(..., ge=0.0, le=1.0)
    last_crop: str
    season: int = Field(..., ge=0, le=10)
    weather: Literal["rainy", "normal", "drought"]
    groundwater: float = Field(..., ge=0.0, le=1.0)
    budget: float


class Action(SdkAction):
    crop: Literal["rice", "wheat", "none"]
    fertilizer: float = Field(..., ge=0.0, le=1.0)
    irrigation: float = Field(..., ge=0.0, le=1.0)


class AgriState(SdkState):
    task_id: str = "hard"
    reward_history: List[float] = Field(default_factory=list)
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    soil_trace: List[float] = Field(default_factory=list)
    nitrogen_trace: List[float] = Field(default_factory=list)
    budget_trace: List[float] = Field(default_factory=list)
    penalty_history: List[float] = Field(default_factory=list)


class Reward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)


class StepInfo(BaseModel):
    yield_score: float
    soil_health: float
    water_used: float
    budget_remaining: float
    penalties: dict

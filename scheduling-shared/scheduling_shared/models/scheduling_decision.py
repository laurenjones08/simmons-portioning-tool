from __future__ import annotations

from datetime import date
from typing import Optional

from bson import ObjectId
from pydantic import BaseModel, Field


class SchedulingDecisionBase(BaseModel):
    mix_id: str = Field(..., alias="mixId", min_length=1, max_length=100)
    line_id: str = Field(..., alias="lineId", min_length=1, max_length=100)
    decision_date: date = Field(..., alias="date")
    duration: float = Field(..., alias="duration", gt=0.0)
    lbs_produced: float = Field(..., alias="lbsProduced", ge=0.0)
    upgrade_pct: float = Field(..., alias="upgradePct", ge=0.0)

    model_config = {"populate_by_name": True}


class SchedulingDecisionCreate(SchedulingDecisionBase):
    pass


class SchedulingDecisionUpdate(SchedulingDecisionBase):
    pass


class SchedulingDecision(SchedulingDecisionBase):
    scheduling_decision_id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "_id": "65f0c8fd6fb6bd463e25d4b7",
                "mixId": "mix-001",
                "lineId": "DSI884",
                "date": "2026-04-15",
                "duration": 6.5,
                "lbsProduced": 2400.0,
                "upgradePct": 0.0875,
            }
        },
    }


class SchedulingDecisionSearchCriteria(BaseModel):
    mix_id: Optional[str] = Field(None, alias="mixId", min_length=1, max_length=100)
    line_id: Optional[str] = Field(None, alias="lineId", min_length=1, max_length=100)
    decision_date: Optional[date] = Field(None, alias="date")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {"example": {"mixId": "mix-001", "lineId": "DSI884"}},
    }

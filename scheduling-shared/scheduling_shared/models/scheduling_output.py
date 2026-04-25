from __future__ import annotations

from datetime import date
from typing import Optional

from bson import ObjectId
from pydantic import BaseModel, Field


class SchedulingOutputBase(BaseModel):
    decision_id: str = Field(..., alias="decisionId", min_length=1, max_length=100)
    sku_id: str = Field(..., alias="skuId", min_length=1, max_length=100)
    output_date: date = Field(..., alias="date")
    batch_upgrade_pct: float = Field(..., alias="batchUpgradePct", ge=0.0)
    lbs_produced: float = Field(..., alias="lbsProduced", ge=0.0)
    short_term_contract_lbs: float = Field(..., alias="shortTermContractLbs", ge=0.0)
    long_term_contract_lbs: float = Field(..., alias="longTermContractLbs", ge=0.0)

    model_config = {"populate_by_name": True}


class SchedulingOutputCreate(SchedulingOutputBase):
    pass


class SchedulingOutputUpdate(SchedulingOutputBase):
    pass


class SchedulingOutput(SchedulingOutputBase):
    scheduling_output_id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "_id": "65f0c8fd6fb6bd463e25d4b7",
                "decisionId": "decision-001",
                "skuId": "50624",
                "date": "2026-04-15",
                "batchUpgradePct": 0.0875,
                "lbsProduced": 2400.0,
                "shortTermContractLbs": 1200.0,
                "longTermContractLbs": 2600.0,
            }
        },
    }


class SchedulingOutputSearchCriteria(BaseModel):
    decision_id: Optional[str] = Field(None, alias="decisionId", min_length=1, max_length=100)
    sku_id: Optional[str] = Field(None, alias="skuId", min_length=1, max_length=100)
    output_date: Optional[date] = Field(None, alias="date")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {"example": {"decisionId": "decision-001", "skuId": "50624"}},
    }

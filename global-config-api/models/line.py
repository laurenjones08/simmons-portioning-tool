"""Data models for production line management."""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class LineType(str, Enum):
    """Stable production line capability types."""

    DB20 = "DB20"
    DSI884 = "DSI884"
    DSI888 = "DSI888"


class LineBase(BaseModel):
    """Shared fields for production line documents."""

    friendly_name: str = Field(..., alias="friendlyName", min_length=1, max_length=100)
    line_type: LineType = Field(..., alias="lineType")
    plant: str = Field(..., alias="plant", min_length=1, max_length=100)
    hours_of_labor_available_per_shift: float = Field(
        ...,
        alias="hoursOfLaborAvailablePerShift",
        gt=0,
    )
    line_throughput: Optional[float] = Field(
        None,
        alias="lineThroughput",
        ge=0.0,
        description="Total line throughput capacity in lbs/hour",
    )
    permitted_cut_strategy_ids: List[str] = Field(
        default_factory=list,
        alias="permittedCutStrategyIds",
    )
    is_active: bool = Field(True, alias="isActive")

    @field_validator("friendly_name", "plant")
    @classmethod
    def strip_text_fields(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be blank")
        return cleaned

    @field_validator("permitted_cut_strategy_ids", mode="before")
    @classmethod
    def normalize_strategy_ids(cls, values):
        if values is None:
            return []
        if not isinstance(values, list):
            return values

        normalized = []
        seen = set()
        for value in values:
            strategy_id = str(value).strip()
            if not strategy_id:
                raise ValueError("permittedCutStrategyIds must not contain blank values")
            if strategy_id in seen:
                raise ValueError("permittedCutStrategyIds must not contain duplicates")
            seen.add(strategy_id)
            normalized.append(strategy_id)
        return normalized

    model_config = {
        "populate_by_name": True,
    }


class LineCreate(LineBase):
    """Request payload for creating a line."""

    line_id: str = Field(..., alias="lineId", min_length=1, max_length=100)

    @field_validator("line_id")
    @classmethod
    def strip_line_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("lineId must not be blank")
        return cleaned

    model_config = {
        "populate_by_name": True,
    }


class LineUpdate(LineBase):
    """Request payload for replacing an existing line."""

    model_config = {
        "populate_by_name": True,
    }


class Line(LineBase):
    """Production line entity stored in MongoDB."""

    line_id: str = Field(..., alias="lineId", min_length=1, max_length=100)
    created_at: datetime = Field(..., alias="createdAt")
    updated_at: datetime = Field(..., alias="updatedAt")

    model_config = {
        "populate_by_name": True,
    }

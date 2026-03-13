from typing import List, Optional

from bson import ObjectId
from pydantic import BaseModel, Field, field_validator

from models.mix import MfgType
from models.partCode import PartCode


class CutStrategyBase(BaseModel):
    """Base fields for cut strategy documents."""

    name: str = Field(..., alias="name", min_length=1, max_length=100)
    description: Optional[str] = Field(None, alias="description", max_length=500)
    mfg_type: MfgType = Field(..., alias="mfgType")
    has_nugget: bool = Field(..., alias="hasNugget")
    belt_speed: float = Field(..., alias="beltSpeed", ge=0.0)
    parts: List[PartCode] = Field(..., alias="parts", min_length=1)

    @field_validator("parts")
    @classmethod
    def validate_parts_unique(cls, values: List[PartCode]) -> List[PartCode]:
        if len(set(values)) != len(values):
            raise ValueError("parts must not contain duplicates")
        return values

    model_config = {
        "populate_by_name": True,
    }


class CutStrategyCreate(CutStrategyBase):
    """Request payload for creating a cut strategy."""

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "name": "DSI Standard SB",
                "description": "Primary DSI strategy for SB birds",
                "mfgType": "DSI",
                "hasNugget": True,
                "beltSpeed": 1.2,
                "parts": ["D", "R", "M"],
            }
        },
    }


class CutStrategyUpdate(CutStrategyBase):
    """Request payload for replacing an existing cut strategy."""

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "name": "DSI Standard SB",
                "description": "Updated strategy",
                "mfgType": "DSI",
                "hasNugget": False,
                "beltSpeed": 1.1,
                "parts": ["D", "R"],
            }
        },
    }


class CutStrategy(CutStrategyBase):
    """Cut strategy entity stored in MongoDB."""

    strategy_id: str = Field(
        default_factory=lambda: str(ObjectId()),
        alias="_id",
        description="Unique Cut Strategy ObjectId",
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "_id": "65f0c8fd6fb6bd463e25d4b7",
                "name": "DSI Standard SB",
                "description": "Primary DSI strategy for SB birds",
                "mfgType": "DSI",
                "hasNugget": True,
                "beltSpeed": 1.2,
                "parts": ["D", "R", "M"],
            }
        },
    }


class CutStrategySearchCriteria(BaseModel):
    """Optional filters for searching cut strategies."""

    name: Optional[str] = Field(None, alias="name", min_length=1)
    mfg_type: Optional[MfgType] = Field(None, alias="mfgType")
    has_nugget: Optional[bool] = Field(None, alias="hasNugget")
    includes_part: Optional[PartCode] = Field(None, alias="includesPart")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "mfgType": "DSI",
                "includesPart": "D",
            }
        },
    }

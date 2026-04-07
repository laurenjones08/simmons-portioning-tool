from typing import List, Optional

from bson import ObjectId
from pydantic import AliasChoices, BaseModel, Field, field_validator

from .part_code import PartCode
from .mix import LineType


class CutStrategyBase(BaseModel):
    """Base fields for cut strategy documents."""

    name: str = Field(..., alias="name", min_length=1, max_length=100)
    description: Optional[str] = Field(None, alias="description", max_length=500)
    line_type: LineType = Field(
        ...,
        alias="lineType",
        validation_alias=AliasChoices("lineType", "mfgType"),
        serialization_alias="lineType",
    )
    has_nugget: bool = Field(..., alias="hasNugget")
    belt_speed: float = Field(..., alias="beltSpeed", ge=0.0) # Belt speed in feet per hour (FPH)
    parts: List[PartCode] = Field(..., alias="parts", min_length=1)

    @field_validator("parts", mode="before")
    @classmethod
    def normalize_parts(cls, values):
        """Normalize incoming part codes before enum validation."""
        if isinstance(values, list):
            normalized = []
            for value in values:
                if isinstance(value, str):
                    normalized.append(value.strip().upper())
                else:
                    normalized.append(value)
            return normalized
        return values

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
                "name": "DSI 2 for 1",
                "description": "DSI 2 filet strategy",
                "lineType": "DSI884",
                "hasNugget": False,
                "beltSpeed": 350.0,
                "parts": ["D", "R"],
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
                "lineType": "DSI884",
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
                "lineType": "DSI884",
                "hasNugget": True,
                "beltSpeed": 1.2,
                "parts": ["D", "R", "M"],
            }
        },
    }


class CutStrategySearchCriteria(BaseModel):
    """Optional filters for searching cut strategies."""

    name: Optional[str] = Field(None, alias="name", min_length=1)
    line_type: Optional[LineType] = Field(
        None,
        alias="lineType",
        validation_alias=AliasChoices("lineType", "mfgType"),
        serialization_alias="lineType",
    )
    has_nugget: Optional[bool] = Field(None, alias="hasNugget")
    includes_part: Optional[PartCode] = Field(None, alias="includesPart")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "lineType": "DSI884",
                "includesPart": "D",
            }
        },
    }

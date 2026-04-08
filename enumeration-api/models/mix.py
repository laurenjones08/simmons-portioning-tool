"""MIX data model used for portioning decision inputs."""

from enum import Enum
from typing import Dict, Optional, List

from bson import ObjectId
from pydantic import BaseModel, Field, field_validator, model_validator


class LineType(str, Enum):
    """Supported manufacturing line types for mix execution."""

    DSI884 = "DSI884"
    DSI888 = "DSI888"
    DB20 = "DB20"


MfgType = LineType

class BirdSize(str, Enum):
    """Supported manufacturing line types for mix execution."""

    SB = "SB"
    BB = "BB"
    ALL = "ALL"


class MixBase(BaseModel):
    """Base fields shared by mix create/update/read models."""

    skus: Dict[str, str] = Field(
        ...,
        description="Map of SKU trade number to Part ID, e.g. {'123': 'A', '345': 'B'}",
        min_length=1,
    )
    # Helper array of SKU trade numbers (keys from `skus`) to support indexing and queries
    # in Mongo: store as `skuKeys` for easier querying by SKU presence (e.g. { skuKeys: "123" }).
    sku_keys: Optional[List[str]] = Field(None, alias="skuKeys", description="List of SKU trade numbers present in this mix", min_length=0)
    includes_fds: bool = Field(..., alias="includesFDS")
    includes_rtl: bool = Field(..., alias="includesRTL")
    includes_nug: bool = Field(..., alias="includesNug")
    nugget_target_weight: Optional[float] = Field(
        None,
        alias="nuggetTargetWeight",
        ge=0.0,
        description="Target weight of one nugget; required (> 0) only when includesNug is true",
    )
    num_fillets: int = Field(..., alias="numFillets", ge=0)
    fillet_weight: float = Field(..., alias="filletWeight", ge=0.0)
    mfg_type: MfgType = Field(..., alias="mfgType")
    plant: str = Field(..., alias="reqPlant", min_length=1, max_length=100)
    bird_size: BirdSize = Field(..., alias="reqBirdSize")
    cut_strategy_id: str = Field(..., alias="cutStrategyID", min_length=1, max_length=100)
    belt_speed: float = Field(..., alias="beltSpeed", ge=0.0)

    @field_validator("skus")
    @classmethod
    def validate_skus_map(cls, value: Dict[str, str]) -> Dict[str, str]:
        for sku_trade_number, part_id in value.items():
            if not str(sku_trade_number).strip():
                raise ValueError("SKU trade number keys must be non-empty")
            if not str(part_id).strip():
                raise ValueError("Part ID values must be non-empty")
        return value

    @model_validator(mode="after")
    def validate_nugget_weight_rules(self):
        if self.includes_nug:
            if self.nugget_target_weight is None or self.nugget_target_weight <= 0:
                raise ValueError("nuggetTargetWeight must be > 0 when includesNug is true")
        else:
            if self.nugget_target_weight is not None:
                raise ValueError("nuggetTargetWeight must be null when includesNug is false")
        # Populate sku_keys from the skus map so we can index/search mixes by SKU trade number.
        try:
            # Avoid overwriting if sku_keys was explicitly provided by the caller
            if getattr(self, "sku_keys", None) in (None, []):
                object.__setattr__(self, "sku_keys", list(self.skus.keys()))
        except Exception:
            # If skus isn't a dict for some reason, skip population; validation elsewhere will catch it.
            pass
        return self

    model_config = {
        "populate_by_name": True,
    }


class MixCreate(MixBase):
    """Request payload for creating a mix."""


class MixUpdate(MixBase):
    """Request payload for replacing an existing mix."""


class MIX(MixBase):
    """Represents a set of SKUs used together to make portioning decisions."""

    mix_id: str = Field(
        default_factory=lambda: str(ObjectId()),
        alias="_id",
        description="Unique MIX ObjectId",
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "_id": "65f0c8fd6fb6bd463e25d4b7",
                "skus": {"123": "A", "345": "B", "567": "C"},
                "includesFDS": True,
                "includesRTL": True,
                "includesNug": True,
                "nuggetTargetWeight": 15.5,
                "numFillets": 2,
                "filletWeight": 12.75,
                "mfgType": "DSI",
                "cutStrategyID": "strategy-001",
                "beltSpeed": 1.2,
                "reqPlant": "FSP",
                "reqBirdSize": "SB",
            }
        },
    }


class MixSearchCriteria(BaseModel):
    """Optional filters for searching mixes."""

    includes_fds: Optional[bool] = Field(None, alias="includesFDS")
    includes_rtl: Optional[bool] = Field(None, alias="includesRTL")
    includes_nug: Optional[bool] = Field(None, alias="includesNug")
    plant: Optional[str] = Field(None, alias="reqPlant", min_length=1)
    bird_size: Optional[BirdSize] = Field(None, alias="reqBirdSize")
    mfg_type: Optional[MfgType] = Field(None, alias="mfgType")
    cut_strategy_id: Optional[str] = Field(None, alias="cutStrategyID")
    sku_trade_number: Optional[str] = Field(None, alias="skuTradeNumber", min_length=1)

    model_config = {
        "populate_by_name": True,
    }

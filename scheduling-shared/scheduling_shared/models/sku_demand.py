from __future__ import annotations

from datetime import date
from enum import Enum
from typing import List, Optional

from bson import ObjectId
from pydantic import BaseModel, Field


class DemandType(str, Enum):
    SHORT = "Short"
    LONG = "Long"


class SKUDemandBase(BaseModel):
    sku_id: str = Field(..., alias="skuId", min_length=1, max_length=100)
    demand_value: float = Field(..., alias="demandValue", ge=0.0)
    demand_type: DemandType = Field(..., alias="demandType")
    due_date: date = Field(..., alias="dueDate")

    model_config = {"populate_by_name": True}


class SKUDemandCreate(SKUDemandBase):
    pass


class SKUDemandUpdate(SKUDemandBase):
    pass


class SKUDemand(SKUDemandBase):
    sku_demand_id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "_id": "65f0c8fd6fb6bd463e25d4b7",
                "skuId": "50624",
                "demandValue": 1500.0,
                "demandType": "Short",
                "dueDate": "2026-04-15",
            }
        },
    }


class SKUDemandSearchCriteria(BaseModel):
    sku_id: Optional[str] = Field(None, alias="skuId", min_length=1, max_length=100)
    demand_type: Optional[DemandType] = Field(None, alias="demandType")
    due_date: Optional[date] = Field(None, alias="dueDate")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {"example": {"skuId": "50624", "demandType": "Short"}},
    }


class SKUDemandBulkImportRequest(BaseModel):
    demands: List[SKUDemandCreate] = Field(..., alias="demands", min_length=1)

    model_config = {"populate_by_name": True}


class SKUDemandBulkImportError(BaseModel):
    row_index: int = Field(..., alias="rowIndex", ge=1)
    sku_id: Optional[str] = Field(None, alias="skuId", min_length=1, max_length=100)
    error: str = Field(..., min_length=1)

    model_config = {"populate_by_name": True}


class SKUDemandBulkImportResponse(BaseModel):
    total: int = Field(..., ge=0)
    successful: int = Field(..., ge=0)
    failed: int = Field(..., ge=0)
    errors: List[SKUDemandBulkImportError] = Field(default_factory=list)

    model_config = {"populate_by_name": True}

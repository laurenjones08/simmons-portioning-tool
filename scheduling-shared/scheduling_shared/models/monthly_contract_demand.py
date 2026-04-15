from __future__ import annotations

from typing import Optional

from bson import ObjectId
from pydantic import BaseModel, Field


class MonthlyContractDemandBase(BaseModel):
    sku_id: str = Field(..., alias="skuId", min_length=1, max_length=100)
    year_month: str = Field(
        ...,
        alias="yearMonth",
        min_length=7,
        max_length=7,
        pattern=r"^\d{4}-(0[1-9]|1[0-2])$",
    )
    demand_lbs: float = Field(..., alias="demandLbs", ge=0.0)

    model_config = {"populate_by_name": True}


class MonthlyContractDemandCreate(MonthlyContractDemandBase):
    pass


class MonthlyContractDemandUpdate(MonthlyContractDemandBase):
    pass


class MonthlyContractDemand(MonthlyContractDemandBase):
    monthly_contract_demand_id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "_id": "65f0c8fd6fb6bd463e25d4b7",
                "skuId": "50624",
                "yearMonth": "2026-01",
                "demandLbs": 3000.0,
            }
        },
    }


class MonthlyContractDemandSearchCriteria(BaseModel):
    sku_id: Optional[str] = Field(None, alias="skuId", min_length=1, max_length=100)
    year_month: Optional[str] = Field(
        None,
        alias="yearMonth",
        min_length=7,
        max_length=7,
        pattern=r"^\d{4}-(0[1-9]|1[0-2])$",
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {"example": {"skuId": "50624", "yearMonth": "2026-01"}},
    }


class MonthlyContractDemandBulkSearchRequest(BaseModel):
    sku_ids: list[str] = Field(..., alias="skuIds", min_length=1)
    year_months: list[str] = Field(..., alias="yearMonths", min_length=1)

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "skuIds": ["50624", "50625"],
                "yearMonths": ["2026-01", "2026-02"],
            }
        },
    }


class MonthlyContractDemandBulkImportRequest(BaseModel):
    demands: list[MonthlyContractDemandCreate] = Field(..., alias="demands", min_length=1)

    model_config = {"populate_by_name": True}


class MonthlyContractDemandBulkImportError(BaseModel):
    row_index: int = Field(..., alias="rowIndex", ge=1)
    sku_id: Optional[str] = Field(None, alias="skuId", min_length=1, max_length=100)
    year_month: Optional[str] = Field(None, alias="yearMonth")
    error: str = Field(..., min_length=1)

    model_config = {"populate_by_name": True}


class MonthlyContractDemandBulkImportResponse(BaseModel):
    total: int = Field(..., ge=0)
    successful: int = Field(..., ge=0)
    failed: int = Field(..., ge=0)
    errors: list[MonthlyContractDemandBulkImportError] = Field(default_factory=list)

    model_config = {"populate_by_name": True}

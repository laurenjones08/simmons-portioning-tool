from pydantic import BaseModel, Field

from .part_code import PartCode


class UnitPlanItem(BaseModel):
    sku: str = Field(..., alias="sku", min_length=1)
    units_in_plan: int = Field(..., alias="unitsInPlan", ge=0)
    total_weight_in_plan: float = Field(..., alias="totalWeightInPlan", ge=0.0)
    part_code: PartCode = Field(..., alias="partCode")
    pct_of_total: float | None = Field(
        None,
        alias="pctOfTotal",
        ge=0.0,
        le=100.0,
        description="Percent of total product represented by this unit plan item",
    )

    model_config = {
        "populate_by_name": True,
    }

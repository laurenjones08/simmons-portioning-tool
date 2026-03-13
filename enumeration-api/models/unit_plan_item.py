from pydantic import BaseModel, Field

from models.partCode import PartCode


class UnitPlanItem(BaseModel):
    sku: str = Field(..., alias="sku", min_length=1)
    units_in_plan: int = Field(..., alias="unitsInPlan", ge=0)
    total_weight_in_plan: float = Field(..., alias="totalWeightInPlan", ge=0.0)
    part_code: PartCode = Field(..., alias="partCode")

    model_config = {
        "populate_by_name": True,
    }

from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic import model_validator
from unit_plan_item import UnitPlanItem


# todo class that quantifies upgrade, value, and trim for a mix bucket combo. It also includes the unit plan

class MixMetric(BaseModel):
    # Primary identifiers
    mix_id: str = Field(..., alias="mixId", description="Unique MIX ObjectId")
    bucket_id: str = Field(..., alias="bucketId", description="Unique Bucket ObjectId")

    # Composite primary id (composed of mixId:bucketId). This will be auto-filled when the model
    # is created if not provided; it is used as the canonical primary identifier for this metric.
    # Alias is set to `_id` so that this field maps to MongoDB's primary key.
    metric_id: Optional[str] = Field(None, alias="_id", description="Composite id 'mixId:bucketId' (Mongo _id)")

    upgrade_percentage: float = Field(..., alias="upgradePercentage", ge=0.0, le=100.0, description="Percentage of parts upgraded in this bucket")
    value: float = Field(..., alias="value", ge=0.0, le=100.0, description="Percentage of total value represented by this bucket")
    trim_percentage: float = Field(..., alias="trimPercentage", ge=0.0, le=100.0, description="Percentage of parts trimmed in this bucket")
    unit_plan: List[UnitPlanItem] = Field(..., alias="unitPlan", description="Unit production plan for this bucket mix combo")

    # Optional denormalized array of SKU trade numbers present in the referenced mix.
    # Storing this here allows direct queries like { skuKeys: "123" } against the metrics
    # collection without an additional lookup to the mixes collection. Populate at write time
    # if you prefer denormalization for performance.
    sku_keys: Optional[List[str]] = Field(None, alias="skuKeys", description="Denormalized list of SKU trade numbers present in the mix")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "mixId": "65f0c8fd6fb6bd463e25d4b7",
                "bucketId": "65f0c8fd6fb6bd463e25d4b8",
                "_id": "65f0c8fd6fb6bd463e25d4b7:65f0c8fd6fb6bd463e25d4b8",
                "upgradePercentage": 20.5,
                "valuePercentage": 15.3,
                "trimPercentage": 5.2,
                "unitPlan": [],
                "skuKeys": ["123", "345", "567"],
            }
        },
    }

    @model_validator(mode="after")
    def set_metric_id(self):
        """Ensure metric_id is set to the composite of mix_id and bucket_id.

        If a metricId was provided, validate it matches the composition. Otherwise, populate it.
        """
        composite = f"{self.mix_id}:{self.bucket_id}"
        if self.metric_id is None:
            self.metric_id = composite
        else:
            # If provided, ensure it matches expected composite format
            if self.metric_id != composite:
                raise ValueError("_id must equal '{mixId}:{bucketId}'")
        return self

    @model_validator(mode="after")
    def validate_sku_keys_match_unit_plan(self):
        """Require skuKeys and ensure it exactly matches SKUs present in unitPlan."""
        unit_plan_skus = [str(item.sku).strip() for item in self.unit_plan]
        if any(not sku for sku in unit_plan_skus):
            raise ValueError("unitPlan items must contain non-empty sku values")

        # Expected skuKeys are unique SKUs in the order they first appear in unitPlan.
        expected_skus = list(dict.fromkeys(unit_plan_skus))

        if self.sku_keys is None or len(self.sku_keys) == 0:
            raise ValueError("skuKeys must be explicitly provided and match unitPlan SKUs")

        normalized_sku_keys = [str(sku).strip() for sku in self.sku_keys]
        if any(not sku for sku in normalized_sku_keys):
            raise ValueError("skuKeys must contain non-empty values")
        if len(set(normalized_sku_keys)) != len(normalized_sku_keys):
            raise ValueError("skuKeys must not contain duplicates")

        if normalized_sku_keys != expected_skus:
            raise ValueError("skuKeys must exactly match unitPlan SKUs in first-appearance order")

        self.sku_keys = normalized_sku_keys
        return self

    @property
    def metricId(self) -> Optional[str]:
        """Convenience camelCase accessor used in parts of the codebase.

        This returns the same value as the Mongo `_id` field.
        """
        return self.metric_id

    def to_api_dict(self) -> dict:
        """Return a dict ready for API responses:

        - Uses alias names (mixId, bucketId, etc.)
        - Replaces Mongo `_id` with `metricId` and does not expose `_id` directly
        """
        data = self.model_dump(by_alias=True)
        # If stored as `_id` in the DB, expose it as `metricId` for clients
        if "_id" in data:
            data["metricId"] = data.pop("_id")
        return data


class MixMetricSearchCriteria(BaseModel):
    """Optional filters for searching MixMetric documents."""

    mix_id: Optional[str] = Field(None, alias="mixId")
    bucket_id: Optional[str] = Field(None, alias="bucketId")
    sku_trade_number: Optional[str] = Field(None, alias="skuTradeNumber", min_length=1)

    model_config = {
        "populate_by_name": True,
    }

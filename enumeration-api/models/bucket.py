from typing import Optional

from bson import ObjectId
from pydantic import BaseModel, Field, model_validator


class BucketBase(BaseModel):
    """Base fields for bucket documents."""

    min_weight: float = Field(..., alias="minWeight", gt=0, description="Minimum weight for the bucket")
    target_weight: float = Field(..., alias="targetWeight", gt=0, description="Target weight for the bucket")
    max_weight: float = Field(..., alias="maxWeight", gt=0, description="Maximum weight for the bucket")

    @model_validator(mode="after")
    def validate_weight_rules(self):
        if self.min_weight >= self.max_weight:
            raise ValueError("minWeight must be less than maxWeight")
        if self.target_weight < self.min_weight:
            raise ValueError("targetWeight must be greater than or equal to minWeight")
        if self.target_weight > self.max_weight:
            raise ValueError("targetWeight must be less than or equal to maxWeight")
        return self

    model_config = {
        "populate_by_name": True,
    }


class BucketCreate(BucketBase):
    """Request payload for creating a bucket."""

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "minWeight": 380.0,
                "targetWeight": 380.0,
                "maxWeight": 480.0,
            }
        },
    }


class BucketUpdate(BucketBase):
    """Request payload for replacing an existing bucket."""

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "minWeight": 390.0,
                "targetWeight": 390.0,
                "maxWeight": 490.0,
            }
        },
    }


class Bucket(BucketBase):
    """Represents a bucket of SKUs used for portioning decisions."""

    bucket_id: str = Field(
        default_factory=lambda: str(ObjectId()),
        alias="_id",
        description="Unique Bucket ObjectId",
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "_id": "65f0c8fd6fb6bd463e25d4b7",
                "minWeight": 380.0,
                "maxWeight": 480.0,
            }
        },
    }


class BucketSearchCriteria(BaseModel):
    """Optional filters for searching buckets."""

    min_weight_gte: Optional[float] = Field(None, alias="minWeightGte", gt=0)
    max_weight_lte: Optional[float] = Field(None, alias="maxWeightLte", gt=0)

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "minWeightGte": 350.0,
                "maxWeightLte": 500.0,
            }
        },
    }

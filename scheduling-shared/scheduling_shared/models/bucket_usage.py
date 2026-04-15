from __future__ import annotations

from datetime import date
from typing import Optional

from bson import ObjectId
from pydantic import BaseModel, Field


class BucketUsageBase(BaseModel):
    bucket_id: str = Field(..., alias="bucketId", min_length=1, max_length=100)
    usage_date: date = Field(..., alias="date")
    available_lbs: float = Field(..., alias="availableLbs", ge=0.0)
    utilized_lbs: float = Field(..., alias="utilizedLbs", ge=0.0)

    model_config = {"populate_by_name": True}


class BucketUsageCreate(BucketUsageBase):
    pass


class BucketUsageUpdate(BucketUsageBase):
    pass


class BucketUsage(BaseModel):
    bucket_id: str = Field(..., alias="bucketId", min_length=1, max_length=100)
    usage_date: date = Field(..., alias="date")
    available_lbs: float = Field(..., alias="availableLbs", ge=0.0)
    utilized_lbs: float = Field(..., alias="utilizedLbs", ge=0.0)
    bucket_usage_id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "_id": "65f0c8fd6fb6bd463e25d4b7",
                "bucketId": "bucket-001",
                "date": "2026-04-15",
                "availableLbs": 1200.0,
                "utilizedLbs": 900.0,
            }
        },
    }


class BucketUsageSearchCriteria(BaseModel):
    bucket_id: Optional[str] = Field(None, alias="bucketId", min_length=1, max_length=100)
    usage_date: Optional[date] = Field(None, alias="date")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {"example": {"bucketId": "bucket-001"}},
    }

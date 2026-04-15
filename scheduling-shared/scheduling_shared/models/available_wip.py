from __future__ import annotations

from typing import Optional

from bson import ObjectId
from pydantic import BaseModel, Field


class AvailableWIPBase(BaseModel):
    plant_name: str = Field(..., alias="plantName", min_length=1, max_length=100)
    bucket_id: str = Field(..., alias="bucketId", min_length=1, max_length=100)
    available_lbs: float = Field(..., alias="availableLbs", ge=0.0)

    model_config = {"populate_by_name": True}


class AvailableWIPCreate(AvailableWIPBase):
    pass


class AvailableWIPUpdate(AvailableWIPBase):
    pass


class AvailableWIP(AvailableWIPBase):
    available_wip_id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "_id": "65f0c8fd6fb6bd463e25d4b7",
                "plantName": "FSP",
                "bucketId": "B 0-390",
                "availableLbs": 30468.11,
            }
        },
    }


class AvailableWIPSearchCriteria(BaseModel):
    plant_name: Optional[str] = Field(None, alias="plantName", min_length=1, max_length=100)
    bucket_id: Optional[str] = Field(None, alias="bucketId", min_length=1, max_length=100)

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "plantName": "FSP",
                "bucketId": "B 0-390",
            }
        },
    }


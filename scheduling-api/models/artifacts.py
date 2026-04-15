from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ArtifactFile(BaseModel):
    artifact_name: str = Field(..., alias="artifactName")
    file_name: str = Field(..., alias="fileName")
    bucket: str = Field(..., alias="bucket")
    key: str = Field(..., alias="key")
    download_url: Optional[str] = Field(default=None, alias="downloadUrl")

    model_config = {"populate_by_name": True}

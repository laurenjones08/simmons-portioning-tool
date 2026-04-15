from __future__ import annotations

from typing import List
from urllib.parse import quote

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from config import get_settings
from models.artifacts import ArtifactFile
from storage import read_object_bytes

router = APIRouter()


async def _fetch_worker_artifacts(job_id: str) -> List[ArtifactFile]:
    settings = get_settings()
    url = f"{settings.scheduling_worker_api_url.rstrip('/')}/jobs/{job_id}/artifacts"
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    response.raise_for_status()
    artifacts = response.json()
    return [ArtifactFile(**artifact) for artifact in artifacts]


@router.get("/jobs/{job_id}/artifacts", response_model=List[ArtifactFile], tags=["Jobs"])
async def list_job_artifacts(job_id: str, request: Request) -> List[ArtifactFile]:
    artifacts = await _fetch_worker_artifacts(job_id)
    proxy_base = str(request.base_url).rstrip("/")
    return [
        ArtifactFile(
            artifactName=artifact.artifact_name,
            fileName=artifact.file_name,
            bucket=artifact.bucket,
            key=artifact.key,
            downloadUrl=f"{proxy_base}/jobs/{job_id}/artifacts/{quote(artifact.artifact_name, safe='')}",
        )
        for artifact in artifacts
    ]


@router.get("/jobs/{job_id}/artifacts/{artifact_name}", tags=["Jobs"])
async def download_job_artifact(job_id: str, artifact_name: str) -> Response:
    artifacts = await _fetch_worker_artifacts(job_id)
    matched = next((artifact for artifact in artifacts if artifact.artifact_name == artifact_name), None)
    if matched is None:
        raise HTTPException(status_code=404, detail=f"Artifact {artifact_name} not found for job {job_id}")

    payload = read_object_bytes(matched.bucket, matched.key)
    headers = {
        "Content-Disposition": f'attachment; filename="{matched.file_name}"',
        "X-Artifact-Key": matched.key,
    }
    return Response(content=payload, media_type="text/csv", headers=headers)


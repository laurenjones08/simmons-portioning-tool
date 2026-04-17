from __future__ import annotations

import io
import json
import logging
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from config import get_settings

logger = logging.getLogger(__name__)


@lru_cache()
def get_s3_client():
    import boto3
    from botocore.client import Config

    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.object_store_endpoint_url,
        aws_access_key_id=settings.object_store_access_key_id,
        aws_secret_access_key=settings.object_store_secret_access_key,
        region_name=settings.object_store_region,
        use_ssl=settings.object_store_secure,
        config=Config(signature_version="s3v4"),
    )


def ensure_bucket_exists() -> None:
    from botocore.exceptions import ClientError

    settings = get_settings()
    client = get_s3_client()

    try:
        client.head_bucket(Bucket=settings.object_store_bucket)
    except ClientError:
        logger.info("Creating object store bucket %s", settings.object_store_bucket)
        client.create_bucket(Bucket=settings.object_store_bucket)


def dataframe_to_csv_bytes(dataframe) -> bytes:
    buffer = io.StringIO()
    dataframe.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def json_to_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def upload_csv_artifacts(run_id: str, prefix: str, csv_payloads: Dict[str, bytes]) -> Tuple[str, str, List[str]]:
    settings = get_settings()
    ensure_bucket_exists()
    client = get_s3_client()

    normalized_prefix = prefix.strip("/").replace("\\", "/")
    key_prefix = f"runs/{run_id}"
    if normalized_prefix:
        key_prefix = f"{key_prefix}/{normalized_prefix}"

    uploaded_keys: List[str] = []
    for name, payload in csv_payloads.items():
        key = f"{key_prefix}/{name}.csv"
        client.put_object(
            Bucket=settings.object_store_bucket,
            Key=key,
            Body=payload,
            ContentType="text/csv",
        )
        uploaded_keys.append(key)

    return settings.object_store_bucket, key_prefix, uploaded_keys


def upload_json_artifact(run_id: str, prefix: str, artifact_name: str, payload: Any) -> Tuple[str, str]:
    settings = get_settings()
    ensure_bucket_exists()
    client = get_s3_client()

    normalized_prefix = prefix.strip("/").replace("\\", "/")
    key_prefix = f"runs/{run_id}"
    if normalized_prefix:
        key_prefix = f"{key_prefix}/{normalized_prefix}"

    key = f"{key_prefix}/{artifact_name}.json"
    client.put_object(
        Bucket=settings.object_store_bucket,
        Key=key,
        Body=json_to_bytes(payload),
        ContentType="application/json",
    )
    return settings.object_store_bucket, key


def fetch_json_artifact(bucket: str, key: str) -> Any:
    client = get_s3_client()
    response = client.get_object(Bucket=bucket, Key=key)
    body = response["Body"].read()
    return json.loads(body.decode("utf-8"))


def build_download_url(bucket: str, key: str) -> str:
    client = get_s3_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=60 * 60 * 24,
    )


def upload_short_term_demand_file(file_bytes: bytes, suffix: str) -> str:
    from uuid import uuid4

    settings = get_settings()
    ensure_bucket_exists()
    client = get_s3_client()
    key = f"short-term-uploads/{uuid4().hex}{suffix}"
    content_type = "text/csv" if suffix in {".csv", ".tsv", ".txt"} else "application/octet-stream"
    client.put_object(
        Bucket=settings.object_store_bucket,
        Key=key,
        Body=file_bytes,
        ContentType=content_type,
    )
    return key


def download_object_bytes(key: str) -> bytes:
    settings = get_settings()
    client = get_s3_client()
    response = client.get_object(Bucket=settings.object_store_bucket, Key=key)
    return response["Body"].read()


def delete_object(key: str) -> None:
    settings = get_settings()
    client = get_s3_client()
    client.delete_object(Bucket=settings.object_store_bucket, Key=key)

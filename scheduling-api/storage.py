from __future__ import annotations

import io
from functools import lru_cache

from config import get_settings


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


def read_object_bytes(bucket: str, key: str) -> bytes:
    client = get_s3_client()
    response = client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


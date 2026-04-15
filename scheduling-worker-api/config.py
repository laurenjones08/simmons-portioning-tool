from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mongodb_url: str = Field(default="mongodb://root:example@mongodb:27017")
    mongodb_database: str = Field(default="scheduling_db")
    service_name: str = Field(default="scheduling-worker-api")
    root_path: str = Field(default="/api/scheduling-worker")
    enumeration_api_url: str = Field(default="http://api-gateway:80/api/enumeration", alias="ENUMERATION_API_URL")
    object_store_endpoint_url: str = Field(default="http://minio:9000", alias="OBJECT_STORE_ENDPOINT_URL")
    object_store_access_key_id: str = Field(default="minioadmin", alias="OBJECT_STORE_ACCESS_KEY_ID")
    object_store_secret_access_key: str = Field(default="minioadmin", alias="OBJECT_STORE_SECRET_ACCESS_KEY")
    object_store_bucket: str = Field(default="scheduling-artifacts", alias="OBJECT_STORE_BUCKET")
    object_store_region: str = Field(default="us-east-1", alias="OBJECT_STORE_REGION")
    object_store_secure: bool = Field(default=False, alias="OBJECT_STORE_SECURE")
    object_store_public_base_url: str = Field(default="http://localhost:9000", alias="OBJECT_STORE_PUBLIC_BASE_URL")
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()

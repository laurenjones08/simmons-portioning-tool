from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mongodb_url: str = Field(default="mongodb://root:example@mongodb:27017")
    mongodb_database: str = Field(default="scheduling_db")
    service_name: str = Field(default="scheduling-api")
    jaeger_agent_host: str = Field(default="jaeger")
    jaeger_agent_port: int = Field(default=6831)
    scheduling_worker_api_url: str = Field(default="http://scheduling-worker-api:8004")
    object_store_endpoint_url: str = Field(default="http://minio:9000")
    object_store_access_key_id: str = Field(default="minioadmin")
    object_store_secret_access_key: str = Field(default="minioadmin")
    object_store_bucket: str = Field(default="scheduling-artifacts")
    object_store_region: str = Field(default="us-east-1")
    object_store_secure: bool = Field(default=False)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()

"""
Configuration for the Enumeration Worker API.
Loaded from environment variables (or .env file).
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mongodb_url: str = Field(
        default="mongodb://root:example@mongodb:27017",
        description="MongoDB connection URL",
    )
    mongodb_database: str = Field(
        default="enumeration_db",
        description="MongoDB database name (same DB as enumeration-api)",
    )
    service_name: str = Field(
        default="enumeration-worker-api",
        description="Service name for logging",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()

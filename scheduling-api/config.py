from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mongodb_url: str = Field(default="mongodb://root:example@mongodb:27017")
    mongodb_database: str = Field(default="scheduling_db")
    service_name: str = Field(default="scheduling-api")
    jaeger_agent_host: str = Field(default="jaeger")
    jaeger_agent_port: int = Field(default=6831)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()


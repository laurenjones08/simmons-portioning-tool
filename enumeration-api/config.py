"""
Configuration module for the Enumeration API.

This module uses Pydantic Settings to load and validate configuration from environment
variables. Pydantic Settings provides automatic validation, type conversion, and
default values for configuration parameters.

The Settings class inherits from BaseSettings, which automatically:
1. Loads values from environment variables
2. Loads values from .env file if present
3. Validates types and constraints
4. Provides IDE autocomplete for configuration fields

Usage:
    from config import get_settings
    
    settings = get_settings()
    print(settings.mongodb_url)  # Access configuration values
"""

from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    This class defines all configuration parameters needed by the Enumeration API.
    Each field can be set via environment variables (uppercase) or in a .env file.
    
    Attributes:
        mongodb_url: MongoDB connection string (e.g., mongodb://user:pass@host:port)
        mongodb_database: Name of the MongoDB database to use
        service_name: Name of this service for logging and tracing
        jaeger_agent_host: Hostname of the Jaeger agent for distributed tracing
        jaeger_agent_port: Port number of the Jaeger agent
    
    Example:
        # Set via environment variable:
        # export MONGODB_URL=mongodb://localhost:27017
        
        # Or in .env file:
        # MONGODB_URL=mongodb://localhost:27017
        
        settings = Settings()
        print(settings.mongodb_url)
    """
    
    # MongoDB Configuration
    # The Field() function provides additional metadata and validation
    mongodb_url: str = Field(
        default="mongodb://root:example@mongodb:27017",
        description="MongoDB connection URL with credentials and host"
    )
    
    mongodb_database: str = Field(
        default="enumeration_db",
        description="Name of the MongoDB database for SKU data"
    )
    
    # Service Configuration
    service_name: str = Field(
        default="enumeration-api",
        description="Service name for logging and distributed tracing"
    )
    
    # Jaeger Tracing Configuration
    jaeger_agent_host: str = Field(
        default="jaeger",
        description="Hostname of the Jaeger agent for trace export"
    )
    
    jaeger_agent_port: int = Field(
        default=6831,
        description="Port number of the Jaeger agent"
    )
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    This function uses @lru_cache to ensure Settings is instantiated only once
    and reused across the application. This is important because:
    1. Settings instantiation reads from disk (.env file)
    2. We want consistent configuration throughout the app lifecycle
    3. It improves performance by avoiding repeated file reads
    
    The cache is cleared only when the application restarts.
    
    Returns:
        Settings: Singleton settings instance
        
    Example:
        # First call creates and caches the instance
        settings = get_settings()
        
        # Subsequent calls return the same cached instance
        same_settings = get_settings()
        assert settings is same_settings  # True
    """
    return Settings()

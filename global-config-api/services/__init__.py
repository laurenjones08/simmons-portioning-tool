"""
Services package for Global Config API.

This package contains the business logic layer for configuration management.
Services sit between routers (API endpoints) and repositories (data access),
providing validation, error handling, and business rules.
"""

from services.config_service import ConfigService

__all__ = ["ConfigService"]

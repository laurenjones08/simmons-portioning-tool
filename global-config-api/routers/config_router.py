"""
Configuration Router - API Endpoints for Configuration Management.

This module implements the FastAPI router layer for configuration operations.
The router layer handles HTTP requests/responses, request validation, and
delegates business logic to the service layer.

Endpoints:
- GET /health: Health check endpoint
- GET /config/{key}: Retrieve a single configuration by key
- PUT /config/{key}: Create or update a configuration
- GET /config: Retrieve all configurations
- POST /config/batch: Batch update multiple configurations

Architecture:
Router Layer (this file) → Service Layer → Repository Layer → MongoDB

The router uses FastAPI's dependency injection to get service instances,
making it easy to test and maintain.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.database import Database

from models.config import Config, ConfigUpdate, BatchUpdateRequest, BatchUpdateResult
from services.config_service import ConfigService
from repositories.config_repository import ConfigRepository
from database import get_database


# Create FastAPI router
# The router groups related endpoints and can be included in the main app
# with a common prefix and tags for OpenAPI documentation
router = APIRouter()


def get_config_service(db: Database = Depends(get_database)) -> ConfigService:
    """
    Dependency injection function for ConfigService.
    
    This function creates a ConfigService instance with its dependencies
    (ConfigRepository) and is used by FastAPI's dependency injection system.
    
    The dependency chain:
    1. get_database() provides Database instance
    2. ConfigRepository is created with the database
    3. ConfigService is created with the repository
    4. Service is injected into route handlers
    
    Args:
        db: MongoDB database instance (injected by FastAPI)
        
    Returns:
        ConfigService: Configured service instance
        
    Example:
        @router.get("/config/{key}")
        async def get_config(
            key: str,
            service: ConfigService = Depends(get_config_service)
        ):
            # service is automatically injected
            return service.get_config_by_key(key)
    """
    repository = ConfigRepository(db)
    return ConfigService(repository)


@router.get(
    "/health",
    tags=["Health"],
    summary="Health check endpoint",
    description="Returns the health status of the configuration service",
    response_description="Service health status",
    status_code=status.HTTP_200_OK
)
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    This endpoint provides a simple way to verify that the configuration
    service is running and responsive. It's used by:
    - Docker health checks
    - Kubernetes liveness/readiness probes
    - Load balancers to determine service availability
    - Monitoring systems to track uptime
    
    Returns:
        dict: Health status message
        
    Example Response:
        {
            "status": "healthy"
        }
    
    HTTP Status Codes:
        200 OK: Service is healthy and operational
    """
    return {"status": "healthy"}


@router.get(
    "/{key}",
    response_model=Config,
    tags=["Configuration"],
    summary="Get configuration by key",
    description="Retrieve a single configuration value by its unique key identifier",
    response_description="Configuration document with key, value, type, and metadata",
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Configuration found and returned successfully",
            "content": {
                "application/json": {
                    "example": {
                        "key": "enumeration.defaultMaxTrim",
                        "value": 2,
                        "valueType": "int",
                        "description": "Default max trim allowed",
                        "updatedAt": "2024-03-08T10:30:00Z",
                        "minValue": 0,
                        "maxValue": 100
                    }
                }
            }
        },
        404: {
            "description": "Configuration not found",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Configuration with key 'invalid.key' not found"
                    }
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Database connection error"
                    }
                }
            }
        }
    }
)
async def get_config(
    key: str,
    service: ConfigService = Depends(get_config_service)
):
    """
    Retrieve a configuration by its key.
    
    This endpoint fetches a single configuration document from the database
    using the provided key as the unique identifier. The key is used as the
    MongoDB _id field for fast primary key lookups.
    
    Args:
        key: The unique configuration key identifier (path parameter)
        service: ConfigService instance (injected by FastAPI)
        
    Returns:
        Config: Configuration document with all fields
        
    Raises:
        HTTPException 404: If configuration with the specified key doesn't exist
        HTTPException 500: If database connection or other unexpected error occurs
        
    Example Request:
        GET /config/enumeration.defaultMaxTrim
        
    Example Response (200 OK):
        {
            "key": "enumeration.defaultMaxTrim",
            "value": 2,
            "valueType": "int",
            "description": "Default max trim allowed",
            "updatedAt": "2024-03-08T10:30:00Z",
            "minValue": 0,
            "maxValue": 100
        }
        
    Example Error Response (404 Not Found):
        {
            "detail": "Configuration with key 'invalid.key' not found"
        }
    
    Business Logic:
    - Performs primary key lookup in MongoDB (very fast)
    - Returns complete configuration document if found
    - Returns 404 if configuration doesn't exist
    
    **Validates: Requirements 10.1, 10.2, 10.3**
    """
    try:
        # Fetch configuration from service layer
        config = service.get_config_by_key(key)
        
        # Return 404 if not found
        if config is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Configuration with key '{key}' not found"
            )
        
        return config
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 404)
        raise
    except Exception as e:
        # Log and return 500 for unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection error"
        )


@router.put(
    "/{key}",
    response_model=Config,
    tags=["Configuration"],
    summary="Create or update configuration",
    description="Create a new configuration or update an existing one by key",
    response_description="Updated configuration document",
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Configuration created or updated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "key": "enumeration.defaultMaxTrim",
                        "value": 5,
                        "valueType": "int",
                        "description": "Default max trim allowed",
                        "updatedAt": "2024-03-08T10:35:00Z",
                        "minValue": 0,
                        "maxValue": 100
                    }
                }
            }
        },
        422: {
            "description": "Validation error",
            "content": {
                "application/json": {
                    "examples": {
                        "type_mismatch": {
                            "summary": "Value type mismatch",
                            "value": {
                                "detail": "Value must be of type int, got str"
                            }
                        },
                        "range_error": {
                            "summary": "Value out of range",
                            "value": {
                                "detail": "Value 150 exceeds maximum 100"
                            }
                        }
                    }
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Database connection error"
                    }
                }
            }
        }
    }
)
async def update_config(
    key: str,
    update: ConfigUpdate,
    service: ConfigService = Depends(get_config_service)
):
    """
    Create or update a configuration.
    
    This endpoint implements the upsert pattern - it will create a new
    configuration if the key doesn't exist, or update an existing one if it does.
    The updatedAt timestamp is automatically set to the current time.
    
    Validation:
    - Value must match the type specified by valueType
    - For numeric types (int, float), value must be within minValue/maxValue if specified
    - All required fields must be present
    - maxValue must be greater than minValue if both are specified
    
    Args:
        key: The configuration key identifier (path parameter)
        update: Configuration update data (request body)
        service: ConfigService instance (injected by FastAPI)
        
    Returns:
        Config: Updated configuration document with current timestamp
        
    Raises:
        HTTPException 422: If validation fails (type mismatch, range error, etc.)
        HTTPException 500: If database connection or other unexpected error occurs
        
    Example Request:
        PUT /config/enumeration.defaultMaxTrim
        Content-Type: application/json
        
        {
            "value": 5,
            "valueType": "int",
            "description": "Default max trim allowed",
            "minValue": 0,
            "maxValue": 100
        }
        
    Example Response (200 OK):
        {
            "key": "enumeration.defaultMaxTrim",
            "value": 5,
            "valueType": "int",
            "description": "Default max trim allowed",
            "updatedAt": "2024-03-08T10:35:00Z",
            "minValue": 0,
            "maxValue": 100
        }
        
    Example Error Response (422 Unprocessable Entity):
        {
            "detail": "Value must be of type int, got str"
        }
    
    Business Logic:
    - Validates update data using Pydantic models
    - Sets updatedAt timestamp automatically
    - Creates new configuration if key doesn't exist
    - Updates existing configuration if key exists
    - Returns the complete updated configuration
    
    **Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5**
    """
    try:
        # Update configuration via service layer
        # The service handles timestamp setting and validation
        config = service.update_config(key, update)
        return config
        
    except ValueError as e:
        # Validation errors from Pydantic or business logic
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        # Log and return 500 for unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection error"
        )


@router.get(
    "",
    response_model=List[Config],
    tags=["Configuration"],
    summary="Get all configurations",
    description="Retrieve all configuration documents from the database",
    response_description="List of all configuration documents",
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "All configurations returned successfully",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "key": "enumeration.defaultMaxTrim",
                            "value": 2,
                            "valueType": "int",
                            "description": "Default max trim allowed",
                            "updatedAt": "2024-03-08T10:30:00Z",
                            "minValue": 0,
                            "maxValue": 100
                        },
                        {
                            "key": "enumeration.lineEfficiency",
                            "value": 0.85,
                            "valueType": "float",
                            "description": "Production line efficiency factor",
                            "updatedAt": "2024-03-08T09:15:00Z",
                            "minValue": 0.0,
                            "maxValue": 1.0
                        }
                    ]
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Database connection error"
                    }
                }
            }
        }
    }
)
async def get_all_configs(
    service: ConfigService = Depends(get_config_service)
):
    """
    Retrieve all configurations from the database.
    
    This endpoint fetches all configuration documents and returns them as a list.
    It's useful for:
    - Admin dashboards showing all system configurations
    - Configuration export for backup
    - Bulk configuration validation
    - System health checks
    
    Args:
        service: ConfigService instance (injected by FastAPI)
        
    Returns:
        List[Config]: List of all configuration documents
        Returns empty list if no configurations exist
        
    Raises:
        HTTPException 500: If database connection or other unexpected error occurs
        
    Example Request:
        GET /config
        
    Example Response (200 OK):
        [
            {
                "key": "enumeration.defaultMaxTrim",
                "value": 2,
                "valueType": "int",
                "description": "Default max trim allowed",
                "updatedAt": "2024-03-08T10:30:00Z",
                "minValue": 0,
                "maxValue": 100
            },
            {
                "key": "enumeration.lineEfficiency",
                "value": 0.85,
                "valueType": "float",
                "description": "Production line efficiency factor",
                "updatedAt": "2024-03-08T09:15:00Z",
                "minValue": 0.0,
                "maxValue": 1.0
            }
        ]
        
    Example Response (Empty):
        []
    
    Business Logic:
    - Fetches all documents from global_config collection
    - Converts each document to Config model
    - Returns as JSON array
    
    Performance:
    - Configuration collections are typically small (dozens to hundreds)
    - Loading all into memory is acceptable for this use case
    - For very large collections, consider implementing pagination
    
    **Validates: Requirements 12.1, 12.2, 12.3**
    """
    try:
        # Fetch all configurations from service layer
        configs = service.get_all_configs()
        return configs
        
    except Exception as e:
        # Log and return 500 for unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection error"
        )


@router.post(
    "/batch",
    response_model=BatchUpdateResult,
    tags=["Configuration"],
    summary="Batch update configurations",
    description="Update multiple configurations in a single request with all-or-nothing validation",
    response_description="Batch update result with success/failure counts",
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Batch update completed (check result for individual failures)",
            "content": {
                "application/json": {
                    "examples": {
                        "all_success": {
                            "summary": "All updates successful",
                            "value": {
                                "total": 2,
                                "successful": 2,
                                "failed": 0,
                                "errors": []
                            }
                        },
                        "validation_failure": {
                            "summary": "Validation failures",
                            "value": {
                                "total": 3,
                                "successful": 0,
                                "failed": 1,
                                "errors": [
                                    {
                                        "index": 1,
                                        "key": "enumeration.invalidParam",
                                        "error": "Value must be of type int, got str"
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        },
        422: {
            "description": "Request validation error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "configs field is required"
                    }
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Database connection error"
                    }
                }
            }
        }
    }
)
async def batch_update_configs(
    request: BatchUpdateRequest,
    service: ConfigService = Depends(get_config_service)
):
    """
    Update multiple configurations with all-or-nothing validation.
    
    This endpoint implements batch configuration updates with comprehensive
    validation. All configurations are validated before any updates occur,
    ensuring data consistency.
    
    Validation Strategy:
    - Phase 1: Validate all updates (Pydantic validation + business rules)
    - Phase 2: If all valid, apply updates (or skip if validateOnly=true)
    - All-or-nothing: Either all updates succeed or none are applied
    
    This ensures you never end up with partially updated configurations
    that could leave the system in an inconsistent state.
    
    Args:
        request: Batch update request with list of config updates
        service: ConfigService instance (injected by FastAPI)
        
    Returns:
        BatchUpdateResult: Summary with success/failure counts and error details
        
    Raises:
        HTTPException 422: If request body validation fails
        HTTPException 500: If database connection or other unexpected error occurs
        
    Example Request:
        POST /config/batch
        Content-Type: application/json
        
        {
            "configs": [
                {
                    "key": "enumeration.defaultMaxTrim",
                    "update": {
                        "value": 5,
                        "valueType": "int",
                        "description": "Maximum trim allowed",
                        "minValue": 0,
                        "maxValue": 100
                    }
                },
                {
                    "key": "enumeration.lineEfficiency",
                    "update": {
                        "value": 0.85,
                        "valueType": "float",
                        "description": "Production line efficiency factor",
                        "minValue": 0.0,
                        "maxValue": 1.0
                    }
                }
            ],
            "validateOnly": false
        }
        
    Example Response (All Success):
        {
            "total": 2,
            "successful": 2,
            "failed": 0,
            "errors": []
        }
        
    Example Response (Validation Failure):
        {
            "total": 3,
            "successful": 0,
            "failed": 1,
            "errors": [
                {
                    "index": 1,
                    "key": "enumeration.invalidParam",
                    "error": "Value must be of type int, got str"
                }
            ]
        }
    
    Validation-Only Mode:
    - Set validateOnly=true to check if updates are valid without applying them
    - Useful for pre-flight validation in UIs
    - Returns validation results without modifying the database
    
    Business Logic:
    - Validates all configurations first
    - If any validation fails, rejects entire batch (no partial updates)
    - If all valid and not validateOnly, updates all configurations
    - Returns summary with success/failure counts and error details
    
    **Validates: Requirements 18.7, 18.8, 18.9, 18.10**
    """
    try:
        # Perform batch update via service layer
        # The service handles all-or-nothing validation and update logic
        result = service.batch_update(
            request.configs,
            request.validate_only
        )
        return result
        
    except ValueError as e:
        # Request validation errors
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        # Log and return 500 for unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection error"
        )

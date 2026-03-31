"""
Configuration data models and validation.

This module defines Pydantic models for configuration data, including:
- ValueType: Enum for supported configuration value types
- Config: Main configuration document model with field validation
- ConfigUpdate: Model for configuration update requests
- BatchConfigUpdate: Model for individual config updates in batch operations
- BatchUpdateRequest: Model for bulk configuration update requests
- BatchUpdateResult: Model for bulk update operation results

All models use Pydantic for automatic validation and serialization.
"""

from typing import Union, Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
from datetime import datetime


class ValueType(str, Enum):
    """
    Supported configuration value types.
    
    This enum defines the allowed types for configuration values. Each configuration
    must specify its valueType, and the actual value must match that type.
    
    Types:
        INT: Integer values (e.g., 42, -10, 0)
        STRING: Text values (e.g., "hello", "FSP")
        FLOAT: Floating-point numbers (e.g., 3.14, -0.5, 2.0)
        BOOL: Boolean values (True or False)
    """
    INT = "int"
    STRING = "string"
    FLOAT = "float"
    BOOL = "bool"


class Config(BaseModel):
    """
    Configuration data model with comprehensive validation.
    
    This model represents a system configuration parameter with type validation
    and optional numeric range constraints. The key serves as the unique identifier
    and is stored as the MongoDB _id field.
    
    Field Validation:
    - value must match the type specified by valueType
    - For numeric types (int, float), value must be within minValue/maxValue if specified
    - updatedAt is automatically set to current timestamp on updates
    - key is used as the MongoDB _id field
    
    MongoDB Mapping:
    - The _id field in MongoDB equals the key value
    - Field names use snake_case in Python (e.g., value_type, updated_at)
    - MongoDB uses camelCase (e.g., valueType, updatedAt)
    - Pydantic aliases handle the conversion automatically
    
    Example:
        Config(
            key="enumeration.defaultMaxTrim",
            value=2,
            value_type=ValueType.INT,
            description="Default max trim allowed",
            updated_at=datetime.utcnow(),
            min_value=0,
            max_value=100
        )
    """
    
    key: str = Field(
        ...,
        description="Configuration key identifier (used as MongoDB _id)",
        min_length=1,
        max_length=200
    )
    
    value: Union[int, str, float, bool] = Field(
        ...,
        description="Configuration value (type must match valueType)"
    )
    
    value_type: ValueType = Field(
        ...,
        alias="valueType",
        description="Type of the configuration value"
    )
    
    description: str = Field(
        ...,
        description="Human-readable description of the configuration parameter",
        min_length=1,
        max_length=500
    )
    
    updated_at: datetime = Field(
        ...,
        alias="updatedAt",
        description="Last update timestamp in ISO 8601 format"
    )
    
    min_value: Optional[float] = Field(
        None,
        alias="minValue",
        description="Minimum value constraint for numeric types (int, float)"
    )
    
    max_value: Optional[float] = Field(
        None,
        alias="maxValue",
        description="Maximum value constraint for numeric types (int, float)"
    )
    
    @model_validator(mode='after')
    def validate_value_and_range(self):
        """
        Validate that the value matches the declared valueType and is within range constraints.
        
        This validator runs after all fields are populated, ensuring value_type is available.
        
        Type Mapping:
        - INT: Python int
        - STRING: Python str
        - FLOAT: Python int or float (int is acceptable for float type)
        - BOOL: Python bool
        
        Returns:
            The validated model instance
            
        Raises:
            ValueError: If value type doesn't match valueType or is outside range
        """
        # Validate value type matches valueType
        value_type = self.value_type
        v = self.value
        
        # Special case: bool is a subclass of int in Python, so we need to check bool first
        if isinstance(v, bool):
            if value_type != ValueType.BOOL:
                raise ValueError(f'Value must be of type {value_type.value}, got bool')
        elif value_type == ValueType.INT:
            if not isinstance(v, int):
                raise ValueError(f'Value must be of type {value_type.value}, got {type(v).__name__}')
        elif value_type == ValueType.STRING:
            if not isinstance(v, str):
                raise ValueError(f'Value must be of type {value_type.value}, got {type(v).__name__}')
        elif value_type == ValueType.FLOAT:
            if not isinstance(v, (int, float)):
                raise ValueError(f'Value must be of type {value_type.value}, got {type(v).__name__}')
        elif value_type == ValueType.BOOL:
            if not isinstance(v, bool):
                raise ValueError(f'Value must be of type {value_type.value}, got {type(v).__name__}')
        
        # Validate numeric range if applicable (only for int and float, not bool)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if self.min_value is not None and v < self.min_value:
                raise ValueError(f'Value {v} is below minimum {self.min_value}')
            
            if self.max_value is not None and v > self.max_value:
                raise ValueError(f'Value {v} exceeds maximum {self.max_value}')
        
        return self
    
    @field_validator('max_value')
    @classmethod
    def validate_max_greater_than_min(cls, v: Optional[float], info) -> Optional[float]:
        """
        Validate that maxValue is greater than minValue if both are specified.
        
        Args:
            v: The maxValue being validated
            info: Validation context containing other field values
            
        Returns:
            The validated maxValue
            
        Raises:
            ValueError: If maxValue <= minValue
        """
        if v is None:
            return v
        
        min_val = info.data.get('min_value')
        if min_val is not None and v <= min_val:
            raise ValueError(f'maxValue ({v}) must be greater than minValue ({min_val})')
        
        return v
    
    model_config = {
        # Allow population by field name (snake_case) or alias (camelCase)
        "populate_by_name": True,
        # Example JSON for documentation
        "json_schema_extra": {
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


class ConfigUpdate(BaseModel):
    """
    Configuration update request model.
    
    This model is used for creating or updating configuration values. It contains
    all the fields needed to define a configuration except the key (which is provided
    in the URL path) and updatedAt (which is set automatically by the service).
    
    The service layer will:
    1. Validate the update using this model
    2. Set the updatedAt timestamp to the current time
    3. Upsert the configuration in the database
    
    Usage:
        # Create or update a configuration
        ConfigUpdate(
            value=5,
            value_type=ValueType.INT,
            description="Maximum trim allowed",
            min_value=0,
            max_value=100
        )
    """
    
    value: Union[int, str, float, bool] = Field(
        ...,
        description="New configuration value (type must match valueType)"
    )
    
    value_type: ValueType = Field(
        ...,
        alias="valueType",
        description="Type of the configuration value"
    )
    
    description: str = Field(
        ...,
        description="Human-readable description of the configuration parameter",
        min_length=1,
        max_length=500
    )
    
    min_value: Optional[float] = Field(
        None,
        alias="minValue",
        description="Minimum value constraint for numeric types (int, float)"
    )
    
    max_value: Optional[float] = Field(
        None,
        alias="maxValue",
        description="Maximum value constraint for numeric types (int, float)"
    )
    
    @model_validator(mode='after')
    def validate_value_and_range(self):
        """
        Validate that the value matches the declared valueType and is within range constraints.
        
        See Config.validate_value_and_range for detailed documentation.
        """
        # Validate value type matches valueType
        value_type = self.value_type
        v = self.value
        
        # Special case: bool is a subclass of int in Python, so we need to check bool first
        if isinstance(v, bool):
            if value_type != ValueType.BOOL:
                raise ValueError(f'Value must be of type {value_type.value}, got bool')
        elif value_type == ValueType.INT:
            if not isinstance(v, int):
                raise ValueError(f'Value must be of type {value_type.value}, got {type(v).__name__}')
        elif value_type == ValueType.STRING:
            if not isinstance(v, str):
                raise ValueError(f'Value must be of type {value_type.value}, got {type(v).__name__}')
        elif value_type == ValueType.FLOAT:
            if not isinstance(v, (int, float)):
                raise ValueError(f'Value must be of type {value_type.value}, got {type(v).__name__}')
        elif value_type == ValueType.BOOL:
            if not isinstance(v, bool):
                raise ValueError(f'Value must be of type {value_type.value}, got {type(v).__name__}')
        
        # Validate numeric range if applicable (only for int and float, not bool)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if self.min_value is not None and v < self.min_value:
                raise ValueError(f'Value {v} is below minimum {self.min_value}')
            
            if self.max_value is not None and v > self.max_value:
                raise ValueError(f'Value {v} exceeds maximum {self.max_value}')
        
        return self
    
    @field_validator('max_value')
    @classmethod
    def validate_max_greater_than_min(cls, v: Optional[float], info) -> Optional[float]:
        """
        Validate that maxValue is greater than minValue if both are specified.
        """
        if v is None:
            return v
        
        min_val = info.data.get('min_value')
        if min_val is not None and v <= min_val:
            raise ValueError(f'maxValue ({v}) must be greater than minValue ({min_val})')
        
        return v
    
    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "value": 5,
                "valueType": "int",
                "description": "Maximum trim allowed",
                "minValue": 0,
                "maxValue": 100
            }
        }
    }


class BatchConfigUpdate(BaseModel):
    """
    Individual configuration update in a batch operation.
    
    This model pairs a configuration key with its update data for batch operations.
    It's used as part of BatchUpdateRequest to update multiple configurations in
    a single API call.
    
    Fields:
        key: The configuration key to update
        update: The update data (value, type, description, constraints)
    """
    
    key: str = Field(
        ...,
        description="Configuration key identifier",
        min_length=1,
        max_length=200
    )
    
    update: ConfigUpdate = Field(
        ...,
        description="Configuration update data"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "key": "enumeration.defaultMaxTrim",
                "update": {
                    "value": 5,
                    "valueType": "int",
                    "description": "Maximum trim allowed",
                    "minValue": 0,
                    "maxValue": 100
                }
            }
        }
    }


class BatchUpdateRequest(BaseModel):
    """
    Request model for bulk configuration update operations.
    
    This model wraps a list of configuration updates for batch processing, with an
    optional flag to perform validation only without actually updating the data.
    
    Validation Behavior:
    - All configuration updates are validated before any update occurs
    - If any update fails validation, the entire batch is rejected
    - No partial updates - it's all or nothing
    
    Args:
        configs: List of configuration updates to apply
        validate_only: If True, only validate without updating (default: False)
    """
    
    configs: List[BatchConfigUpdate] = Field(
        ...,
        description="List of configuration updates to apply",
        min_length=1
    )
    
    validate_only: bool = Field(
        False,
        alias="validateOnly",
        description="If true, only validate updates without applying them to the database"
    )
    
    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
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
                "validateOnly": False
            }
        }
    }


class BatchUpdateResult(BaseModel):
    """
    Result model for bulk configuration update operations.
    
    This model provides a summary of the batch update operation, including
    counts of successful and failed updates, along with detailed error information
    for any configurations that failed validation or update.
    
    Fields:
        total: Total number of configuration updates in the request
        successful: Number of configurations successfully updated
        failed: Number of configurations that failed validation or update
        errors: List of error details for failed configurations
    
    Error Format:
        Each error is a dictionary containing:
        - index: Position of the failed config in the input list (optional)
        - key: Configuration key that failed
        - error: Description of what went wrong
    """
    
    total: int = Field(
        ...,
        description="Total number of configuration updates in the request",
        ge=0
    )
    
    successful: int = Field(
        ...,
        description="Number of configurations successfully updated",
        ge=0
    )
    
    failed: int = Field(
        ...,
        description="Number of configurations that failed validation or update",
        ge=0
    )
    
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of error details for failed configuration updates"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "total": 5,
                "successful": 4,
                "failed": 1,
                "errors": [
                    {
                        "index": 2,
                        "key": "enumeration.invalidParam",
                        "error": "Value must be of type int, got str"
                    }
                ]
            }
        }
    }

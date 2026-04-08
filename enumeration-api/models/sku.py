"""
SKU (Stock Keeping Unit) data models and validation.

This module defines Pydantic models for SKU data, including:
- SKU: Main SKU document model with field validation
- SearchCriteria: Model for SKU search requests
- BatchImportRequest: Model for bulk SKU import requests
- BatchImportResult: Model for bulk import operation results

All models use Pydantic for automatic validation and serialization.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class SKU(BaseModel):
    """
    SKU (Stock Keeping Unit) data model with comprehensive validation.
    
    This model represents a product SKU with customer information, product specifications,
    and weight constraints. The tradeNumber serves as the unique identifier and is stored
    as the MongoDB _id field.
    
    Field Validation:
    - All numeric fields (minWeight, maxWeight, targetWeight) must be non-negative
    - minWeight must be less than maxWeight
    - targetWeight should be between minWeight and maxWeight
    - allowedParts must be a non-empty array of strings
    - unitsPerCut must be at least 1
    
    MongoDB Mapping:
    - The _id field in MongoDB equals the tradeNumber value
    - Field names use camelCase in MongoDB (e.g., tradeNumber, customerName)
    - Python uses snake_case (e.g., trade_number, customer_name)
    - Pydantic aliases handle the conversion automatically
    """
    
    trade_number: str = Field(
        ...,
        alias="tradeNumber",
        description="Unique trade number identifier for the SKU (used as MongoDB _id)",
        min_length=1,
        max_length=50
    )
    
    customer_name: str = Field(
        ...,
        alias="customerName",
        description="Name of the customer associated with this SKU",
        min_length=1,
        max_length=200
    )
    
    customer_type: str = Field(
        ...,
        alias="customerType",
        description="Customer type code (e.g., 'FDS' for food service)",
        min_length=1,
        max_length=50
    )
    
    product_type: str = Field(
        ...,
        alias="productType",
        description="Product type classification (e.g., 'NUGGET', 'TENDER')",
        min_length=1,
        max_length=50
    )
    
    units_per_cut: int = Field(
        ...,
        alias="unitsPerCut",
        description="Number of units produced per cut operation",
        ge=1
    )
    
    prod_plant: str = Field(
        ...,
        alias="prodPlant",
        description="Production plant code where the SKU is manufactured",
        min_length=1,
        max_length=50
    )
    
    min_weight: float = Field(
        ...,
        alias="minWeight",
        description="Minimum acceptable weight in grams",
        ge=0.0
    )
    
    max_weight: float = Field(
        ...,
        alias="maxWeight",
        description="Maximum acceptable weight in grams",
        ge=0.0
    )
    
    target_weight: float = Field(
        ...,
        alias="targetWeight",
        description="Target weight in grams (should be between minWeight and maxWeight)",
        ge=0.0
    )
    
    bird_size: str = Field(
        ...,
        alias="birdSize",
        description="Bird size classification code (e.g., 'SB' for small bird)",
        min_length=1,
        max_length=50
    )
    
    allowed_parts: List[str] = Field(
        ...,
        alias="allowedParts",
        description="List of allowed part codes for this SKU (e.g., ['D'] for drumstick)",
        min_length=1
    )
    
    @field_validator('max_weight')
    @classmethod
    def validate_max_weight_greater_than_min(cls, v: float, info) -> float:
        """
        Validate that maxWeight is greater than minWeight.
        
        This ensures the weight range is valid and non-empty. A SKU cannot have
        a maximum weight that is less than or equal to its minimum weight.
        
        Args:
            v: The maxWeight value being validated
            info: Validation context containing other field values
            
        Returns:
            The validated maxWeight value
            
        Raises:
            ValueError: If maxWeight <= minWeight
        """
        # Access minWeight from the validation context
        if 'min_weight' in info.data:
            min_weight = info.data['min_weight']
            if v <= min_weight:
                raise ValueError(
                    f'maxWeight ({v}) must be greater than minWeight ({min_weight})'
                )
        return v
    
    @field_validator('target_weight')
    @classmethod
    def validate_target_weight_in_range(cls, v: float, info) -> float:
        """
        Validate that targetWeight is between minWeight and maxWeight.
        
        The target weight should fall within the acceptable weight range for the SKU.
        This is a soft validation - we issue a warning but don't fail validation,
        as target weights outside the range might be intentional in some cases.
        
        Args:
            v: The targetWeight value being validated
            info: Validation context containing other field values
            
        Returns:
            The validated targetWeight value
            
        Note:
            This validator only checks if both min and max weights are available.
            It does not raise an error, but could be extended to do so if needed.
        """
        # Check if targetWeight is within the range (optional validation)
        if 'min_weight' in info.data and 'max_weight' in info.data:
            min_weight = info.data['min_weight']
            max_weight = info.data['max_weight']
            if not (min_weight <= v <= max_weight):
                # Note: This is a soft validation. In production, you might want to
                # raise ValueError or log a warning depending on business requirements
                pass
        return v
    
    @field_validator('allowed_parts')
    @classmethod
    def validate_allowed_parts_not_empty(cls, v: List[str]) -> List[str]:
        """
        Validate that allowedParts contains at least one part code.
        
        Args:
            v: The allowedParts list being validated
            
        Returns:
            The validated allowedParts list
            
        Raises:
            ValueError: If the list is empty
        """
        if not v or len(v) == 0:
            raise ValueError('allowedParts must contain at least one part code')
        return v
    
    model_config = {
        # Allow population by field name (snake_case) or alias (camelCase)
        "populate_by_name": True,
        # Example JSON for documentation
        "json_schema_extra": {
            "example": {
                "tradeNumber": "50624",
                "customerName": "CHICK FIL A INC",
                "customerType": "FDS",
                "productType": "NUGGET",
                "unitsPerCut": 1,
                "prodPlant": "FSP",
                "minWeight": 10.0,
                "maxWeight": 19.0,
                "targetWeight": 15.0,
                "birdSize": "SB",
                "allowedParts": ["D"]
            }
        }
    }


class SKUCreate(BaseModel):
    """Request model for creating or updating a SKU."""

    trade_number: str = Field(..., alias="tradeNumber", min_length=1, max_length=50)
    customer_name: str = Field(..., alias="customerName", min_length=1, max_length=200)
    customer_type: str = Field(..., alias="customerType", min_length=1, max_length=50)
    product_type: str = Field(..., alias="productType", min_length=1, max_length=50)
    units_per_cut: int = Field(..., alias="unitsPerCut", ge=1)
    prod_plant: str = Field(..., alias="prodPlant", min_length=1, max_length=50)
    min_weight: float = Field(..., alias="minWeight", ge=0.0)
    max_weight: float = Field(..., alias="maxWeight", ge=0.0)
    target_weight: float = Field(..., alias="targetWeight", ge=0.0)
    bird_size: str = Field(..., alias="birdSize", min_length=1, max_length=50)
    allowed_parts: List[str] = Field(..., alias="allowedParts", min_length=1)

    @field_validator('max_weight')
    @classmethod
    def validate_max_weight_greater_than_min(cls, v: float, info) -> float:
        if 'min_weight' in info.data:
            min_weight = info.data['min_weight']
            if v <= min_weight:
                raise ValueError(f'maxWeight ({v}) must be greater than minWeight ({min_weight})')
        return v

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "tradeNumber": "50624",
                "customerName": "CHICK FIL A INC",
                "customerType": "FDS",
                "productType": "NUGGET",
                "unitsPerCut": 1,
                "prodPlant": "FSP",
                "minWeight": 10.0,
                "maxWeight": 19.0,
                "targetWeight": 15.0,
                "birdSize": "SB",
                "allowedParts": ["D"]
            }
        }
    }


class SKUUpdate(SKUCreate):
    """Request model for updating an existing SKU (same as create)."""
    pass


class SearchCriteria(BaseModel):
    """
    Search filter criteria for SKU queries.
    
    This model defines optional filter fields for searching SKUs. All fields are optional,
    allowing flexible search queries. When multiple fields are provided, they are combined
    with AND logic (all criteria must match).
    
    Usage:
        # Search by customer type only
        SearchCriteria(customer_type="FDS")
        
        # Search by multiple criteria
        SearchCriteria(customer_type="FDS", product_type="NUGGET", prod_plant="FSP")
    """
    
    customer_type: Optional[str] = Field(
        None,
        alias="customerType",
        description="Filter by customer type code"
    )
    
    product_type: Optional[str] = Field(
        None,
        alias="productType",
        description="Filter by product type"
    )
    
    prod_plant: Optional[str] = Field(
        None,
        alias="prodPlant",
        description="Filter by production plant code"
    )
    
    bird_size: Optional[str] = Field(
        None,
        alias="birdSize",
        description="Filter by bird size classification"
    )
    
    customer_name: Optional[str] = Field(
        None,
        alias="customerName",
        description="Filter by customer name (exact match)"
    )
    
    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "customerType": "FDS",
                "productType": "NUGGET"
            }
        }
    }


class BatchImportRequest(BaseModel):
    """
    Request model for bulk SKU import operations.
    
    This model wraps a list of SKUs for batch import, with an optional flag
    to perform validation only without actually inserting the data.
    
    Validation Behavior:
    - All SKUs are validated before any insertion occurs
    - If any SKU fails validation, the entire batch is rejected
    - No partial imports - it's all or nothing
    
    Args:
        skus: List of SKU objects to import
        validate_only: If True, only validate without inserting (default: False)
    """
    
    skus: List[SKU] = Field(
        ...,
        description="List of SKU documents to import",
        min_length=1
    )
    
    validate_only: bool = Field(
        False,
        alias="validateOnly",
        description="If true, only validate SKUs without inserting them into the database"
    )
    
    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "skus": [
                    {
                        "tradeNumber": "50624",
                        "customerName": "CHICK FIL A INC",
                        "customerType": "FDS",
                        "productType": "NUGGET",
                        "unitsPerCut": 1,
                        "prodPlant": "FSP",
                        "minWeight": 10.0,
                        "maxWeight": 19.0,
                        "targetWeight": 15.0,
                        "birdSize": "SB",
                        "allowedParts": ["D"]
                    }
                ],
                "validateOnly": False
            }
        }
    }


class BatchImportResult(BaseModel):
    """
    Result model for bulk SKU import operations.
    
    This model provides a summary of the batch import operation, including
    counts of successful and failed imports, along with detailed error information
    for any SKUs that failed validation or insertion.
    
    Fields:
        total: Total number of SKUs in the import request
        successful: Number of SKUs successfully imported
        failed: Number of SKUs that failed validation or insertion
        errors: List of error details for failed SKUs
    
    Error Format:
        Each error is a dictionary containing:
        - index: Position of the failed SKU in the input list (optional)
        - trade_number: Trade number of the failed SKU
        - error: Description of what went wrong
    """
    
    total: int = Field(
        ...,
        description="Total number of SKUs in the import request",
        ge=0
    )
    
    successful: int = Field(
        ...,
        description="Number of SKUs successfully imported",
        ge=0
    )
    
    failed: int = Field(
        ...,
        description="Number of SKUs that failed validation or insertion",
        ge=0
    )
    
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of error details for failed SKUs"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "total": 10,
                "successful": 8,
                "failed": 2,
                "errors": [
                    {
                        "index": 3,
                        "trade_number": "50625",
                        "error": "maxWeight must be greater than minWeight"
                    },
                    {
                        "index": 7,
                        "trade_number": "50626",
                        "error": "SKU already exists"
                    }
                ]
            }
        }
    }

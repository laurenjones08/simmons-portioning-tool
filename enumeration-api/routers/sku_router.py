"""
SKU Router - HTTP Endpoint Layer for SKU Operations.

This module implements the FastAPI router for SKU-related endpoints. The router layer
is responsible for:
1. Defining HTTP endpoints and their request/response schemas
2. Handling HTTP-specific concerns (status codes, headers, etc.)
3. Validating request data using Pydantic models
4. Delegating business logic to the service layer
5. Converting service responses to HTTP responses

FastAPI Router Pattern:
- APIRouter: Creates a modular router that can be included in the main app
- Depends(): Dependency injection for services and database connections
- Path parameters: {trade_number} in the URL path
- Request body: Pydantic models for POST/PUT requests
- Response models: Pydantic models for automatic serialization

Architecture:
    HTTP Request → Router (this file) → Service Layer → Repository Layer → MongoDB
    HTTP Response ← Router ← Service Layer ← Repository Layer ← MongoDB

Benefits:
- Separation of concerns: HTTP logic separate from business logic
- Automatic validation: Pydantic validates all inputs
- Auto-generated docs: FastAPI generates OpenAPI/Swagger docs
- Type safety: Full type hints for IDE support
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.database import Database

from models.sku import SKU, SearchCriteria, BatchImportRequest, BatchImportResult
from services.sku_service import SKUService
from repositories.sku_repository import SKURepository
from database import get_database


# Create the router instance
# This router will be included in the main FastAPI app with a prefix (e.g., /skus)
router = APIRouter()


def get_sku_service(db: Database = Depends(get_database)) -> SKUService:
    """
    Dependency injection function for SKUService.
    
    This function creates a SKUService instance with all its dependencies
    (repository, database) properly injected. FastAPI will call this function
    automatically when a route handler declares it as a dependency.
    
    Dependency Injection Flow:
    1. FastAPI sees Depends(get_sku_service) in a route handler
    2. FastAPI calls get_sku_service()
    3. get_sku_service needs a Database, so FastAPI calls get_database()
    4. get_database() yields a Database instance
    5. get_sku_service creates and returns a SKUService
    6. FastAPI injects the SKUService into the route handler
    
    Args:
        db: MongoDB Database instance (injected by FastAPI via get_database)
        
    Returns:
        SKUService: Configured service instance ready for use
        
    Example:
        @router.get("/skus/{trade_number}")
        async def get_sku(
            trade_number: str,
            service: SKUService = Depends(get_sku_service)
        ):
            # service is automatically created and injected
            return service.get_sku_by_trade_number(trade_number)
    """
    repository = SKURepository(db)
    return SKUService(repository)


@router.get(
    "/health",
    tags=["Health"],
    summary="SKU service health check",
    description="Returns the health status of the SKU service",
    response_description="Service health status",
    status_code=status.HTTP_200_OK
)
async def health_check():
    """
    Health check endpoint for the SKU service.
    
    This endpoint is used by monitoring systems, load balancers, and orchestration
    platforms (like Kubernetes) to verify that the service is running and healthy.
    
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
    "/{trade_number}",
    response_model=SKU,
    tags=["SKUs"],
    summary="Get SKU by trade number",
    description="Retrieve a single SKU document by its unique trade number identifier",
    response_description="The SKU document with the specified trade number",
    responses={
        200: {
            "description": "SKU found and returned successfully",
            "content": {
                "application/json": {
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
        },
        404: {
            "description": "SKU not found",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "SKU with trade number 50624 not found"
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
async def get_sku_by_trade_number(
    trade_number: str,
    service: SKUService = Depends(get_sku_service)
):
    """
    Retrieve a SKU by its trade number.
    
    This endpoint fetches a single SKU document from the database using the trade
    number as the unique identifier. The trade number is used as the MongoDB _id
    field, making this a very fast O(1) lookup operation.
    
    Args:
        trade_number: The unique trade number identifier (path parameter)
        service: SKUService instance (injected by FastAPI)
        
    Returns:
        SKU: The SKU document if found
        
    Raises:
        HTTPException 404: If no SKU with the specified trade number exists
        HTTPException 500: If a database error occurs
        
    Example Request:
        GET /skus/50624
        
    Example Response (200 OK):
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
    
    Example Error Response (404 Not Found):
        {
            "detail": "SKU with trade number 50624 not found"
        }
    """
    try:
        # Call service layer to retrieve the SKU
        sku = service.get_sku_by_trade_number(trade_number)
        
        # If SKU not found, return 404
        if sku is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"SKU with trade number {trade_number} not found"
            )
        
        # Return the SKU (FastAPI automatically serializes to JSON)
        return sku
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 404) without modification
        raise
    except Exception as e:
        # Log unexpected errors and return 500
        # In production, use proper logging framework
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving SKU: {str(e)}"
        )


@router.post(
    "/search",
    response_model=List[SKU],
    tags=["SKUs"],
    summary="Search SKUs by criteria",
    description="Search for SKUs matching the specified filter criteria. Multiple criteria are combined with AND logic.",
    response_description="List of SKUs matching the search criteria",
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Search completed successfully (may return empty list if no matches)",
            "content": {
                "application/json": {
                    "example": [
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
                    ]
                }
            }
        },
        422: {
            "description": "Validation error in search criteria",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "customerType"],
                                "msg": "field required",
                                "type": "value_error.missing"
                            }
                        ]
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
async def search_skus(
    criteria: SearchCriteria,
    service: SKUService = Depends(get_sku_service)
):
    """
    Search for SKUs matching filter criteria.
    
    This endpoint allows searching for SKUs using various filter criteria. All
    provided criteria are combined with AND logic (all must match). If no criteria
    are provided, all SKUs are returned.
    
    Search Behavior:
    - All filters use exact matching (not partial or fuzzy matching)
    - Multiple criteria are combined with AND logic
    - Empty criteria returns all SKUs
    - Results are not sorted or paginated
    
    Args:
        criteria: SearchCriteria model with optional filter fields (request body)
        service: SKUService instance (injected by FastAPI)
        
    Returns:
        List[SKU]: List of SKUs matching the criteria (empty list if no matches)
        
    Raises:
        HTTPException 422: If the request body is invalid
        HTTPException 500: If a database error occurs
        
    Example Request:
        POST /skus/search
        Content-Type: application/json
        
        {
            "customerType": "FDS",
            "productType": "NUGGET"
        }
        
    Example Response (200 OK):
        [
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
        ]
    
    Example Empty Result (200 OK):
        []
    """
    try:
        # Call service layer to perform the search
        skus = service.search_skus(criteria)
        
        # Return the list of matching SKUs
        # FastAPI automatically serializes the list to JSON
        return skus
        
    except Exception as e:
        # Log unexpected errors and return 500
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching SKUs: {str(e)}"
        )


@router.post(
    "/batch",
    response_model=BatchImportResult,
    tags=["SKUs"],
    summary="Batch import SKUs",
    description="Import multiple SKUs in a single operation with validation. All SKUs must be valid for any to be imported (all-or-nothing).",
    response_description="Summary of the batch import operation with success/failure counts",
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Batch import completed (check result for success/failure details)",
            "content": {
                "application/json": {
                    "example": {
                        "total": 10,
                        "successful": 10,
                        "failed": 0,
                        "errors": []
                    }
                }
            }
        },
        422: {
            "description": "Validation error in request body",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "skus", 0, "maxWeight"],
                                "msg": "maxWeight must be greater than minWeight",
                                "type": "value_error"
                            }
                        ]
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
async def batch_import_skus(
    request: BatchImportRequest,
    service: SKUService = Depends(get_sku_service)
):
    """
    Import multiple SKUs in a batch operation.
    
    This endpoint allows bulk import of SKUs with comprehensive validation. The
    operation follows an all-or-nothing approach: if any SKU fails validation,
    the entire batch is rejected without inserting any SKUs.
    
    Validation Process:
    1. Pydantic validates each SKU schema (types, constraints, etc.)
    2. Service layer checks for duplicate trade numbers within the batch
    3. Database checks for existing trade numbers (unique constraint)
    4. If all validations pass, all SKUs are inserted atomically
    
    Args:
        request: BatchImportRequest containing list of SKUs and optional validate_only flag
        service: SKUService instance (injected by FastAPI)
        
    Returns:
        BatchImportResult: Summary with total, successful, failed counts and error details
        
    Raises:
        HTTPException 422: If request body is invalid or SKUs fail validation
        HTTPException 500: If a database error occurs
        
    Example Request (Import):
        POST /skus/batch
        Content-Type: application/json
        
        {
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
            "validateOnly": false
        }
        
    Example Request (Validate Only):
        POST /skus/batch
        Content-Type: application/json
        
        {
            "skus": [...],
            "validateOnly": true
        }
        
    Example Response (Success):
        {
            "total": 10,
            "successful": 10,
            "failed": 0,
            "errors": []
        }
        
    Example Response (Partial Failure):
        {
            "total": 10,
            "successful": 0,
            "failed": 10,
            "errors": [
                {
                    "index": 3,
                    "trade_number": "50625",
                    "error": "Duplicate trade number in batch: 50625"
                }
            ]
        }
    """
    try:
        # Call service layer to perform batch import
        result = service.batch_import(request.skus, request.validate_only)
        
        # Return the result summary
        # The result includes success/failure counts and error details
        return result
        
    except Exception as e:
        # Log unexpected errors and return 500
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during batch import: {str(e)}"
        )


@router.get(
    "/export",
    response_model=List[SKU],
    tags=["SKUs"],
    summary="Export all SKUs",
    description="Export all SKU documents from the database in JSON format. Use with caution on large datasets.",
    response_description="List of all SKU documents",
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Export completed successfully",
            "content": {
                "application/json": {
                    "example": [
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
async def export_all_skus(
    service: SKUService = Depends(get_sku_service)
):
    """
    Export all SKUs from the database.
    
    This endpoint retrieves all SKU documents and returns them as a JSON array.
    It's useful for:
    - Backing up SKU data
    - Migrating data to another system
    - Generating reports
    - Data analysis
    
    Performance Considerations:
    - This endpoint loads all SKUs into memory
    - For large datasets (>10,000 SKUs), this may be slow and memory-intensive
    - Consider adding pagination for production use with large datasets
    - Consider adding filters to export subsets of data
    
    Args:
        service: SKUService instance (injected by FastAPI)
        
    Returns:
        List[SKU]: List of all SKU documents in the database
        
    Raises:
        HTTPException 500: If a database error occurs
        
    Example Request:
        GET /skus/export
        
    Example Response (200 OK):
        [
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
            },
            {
                "tradeNumber": "50625",
                ...
            }
        ]
    
    Example Empty Database (200 OK):
        []
    
    Future Enhancements:
    - Add pagination (limit/offset or cursor-based)
    - Add filtering by date range or other criteria
    - Add streaming response for very large datasets
    - Add export format options (CSV, Excel, etc.)
    """
    try:
        # Call service layer to export all SKUs
        skus = service.export_all()
        
        # Return the list of all SKUs
        # FastAPI automatically serializes to JSON
        return skus
        
    except Exception as e:
        # Log unexpected errors and return 500
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error exporting SKUs: {str(e)}"
        )

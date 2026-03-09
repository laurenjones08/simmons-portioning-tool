"""
SKU Service - Business Logic Layer for SKU Operations.

This module implements the service layer for SKU management. The service layer sits
between the router layer (HTTP endpoints) and the repository layer (database access),
providing business logic, validation, and error handling.

Key Responsibilities:
1. Business Logic: Implements domain-specific rules and workflows
2. Validation: Validates data beyond basic schema validation
3. Error Handling: Converts database errors to user-friendly messages
4. Orchestration: Coordinates multiple repository calls if needed
5. Data Transformation: Converts between repository dicts and Pydantic models

Architecture Pattern:
    Router Layer (HTTP) → Service Layer (Business Logic) → Repository Layer (Database)
    
Benefits:
- Separation of concerns: Business logic is isolated from HTTP and database details
- Testability: Easy to test business logic with mocked repositories
- Reusability: Service methods can be called from multiple routes or other services
- Maintainability: Changes to business rules are centralized
"""

from typing import List, Optional, Dict, Any
from pymongo.errors import DuplicateKeyError
from models.sku import SKU, SearchCriteria, BatchImportResult
from repositories.sku_repository import SKURepository


class SKUService:
    """
    Business logic service for SKU operations.
    
    This service provides high-level operations for managing SKUs, including:
    - Retrieving individual SKUs by trade number
    - Searching SKUs by filter criteria
    - Batch importing multiple SKUs with validation
    - Exporting all SKUs
    
    The service handles:
    - Converting between MongoDB documents (dicts) and Pydantic models
    - Validating business rules
    - Providing user-friendly error messages
    - Coordinating batch operations
    
    Attributes:
        repository: SKURepository instance for database access
    """
    
    def __init__(self, repository: SKURepository):
        """
        Initialize the service with a repository dependency.
        
        This follows the dependency injection pattern, making the service
        testable by allowing mock repositories to be injected.
        
        Args:
            repository: SKURepository instance for database operations
            
        Example:
            from database import get_database
            from repositories.sku_repository import SKURepository
            
            db = get_database()
            repo = SKURepository(db)
            service = SKUService(repo)
        """
        self.repository = repository
    
    def get_sku_by_trade_number(self, trade_number: str) -> Optional[SKU]:
        """
        Retrieve a single SKU by its trade number.
        
        This method fetches a SKU from the database and converts it to a
        Pydantic model. If the SKU doesn't exist, it returns None rather
        than raising an exception, allowing the router to decide how to
        handle the missing SKU (typically with a 404 response).
        
        Business Logic:
        - Trade numbers are case-sensitive
        - Returns None for non-existent SKUs (not an error condition)
        
        Args:
            trade_number: The unique trade number identifier
            
        Returns:
            SKU model if found, None if not found
            
        Raises:
            Exception: For database connection errors or unexpected errors
            
        Example:
            service = SKUService(repo)
            
            sku = service.get_sku_by_trade_number("50624")
            if sku:
                print(f"Found: {sku.customer_name}")
            else:
                print("SKU not found")
        """
        try:
            # Call repository to fetch the document
            sku_doc = self.repository.find_by_trade_number(trade_number)
            
            # If not found, return None
            if sku_doc is None:
                return None
            
            # Convert MongoDB document (dict) to Pydantic model
            # Pydantic will validate the data and handle field name conversion
            # (camelCase from MongoDB → snake_case in Python)
            return SKU(**sku_doc)
            
        except Exception as e:
            # Log the error and re-raise for the router to handle
            # In production, use proper logging framework
            raise Exception(f"Error retrieving SKU: {str(e)}")
    
    def search_skus(self, criteria: SearchCriteria) -> List[SKU]:
        """
        Search for SKUs matching the specified filter criteria.
        
        This method converts the SearchCriteria model to a MongoDB query filter
        and returns all matching SKUs. Multiple criteria are combined with AND
        logic (all must match).
        
        Business Logic:
        - Empty criteria returns all SKUs
        - All filters use exact matching (not partial/fuzzy matching)
        - Results are not sorted (returned in natural MongoDB order)
        
        Args:
            criteria: SearchCriteria model with optional filter fields
            
        Returns:
            List of SKU models matching the criteria
            Returns empty list if no matches found
            
        Raises:
            Exception: For database errors
            
        Example:
            # Search by customer type
            criteria = SearchCriteria(customer_type="FDS")
            skus = service.search_skus(criteria)
            
            # Search by multiple criteria
            criteria = SearchCriteria(
                customer_type="FDS",
                product_type="NUGGET",
                prod_plant="FSP"
            )
            skus = service.search_skus(criteria)
            
            print(f"Found {len(skus)} matching SKUs")
        """
        try:
            # Convert SearchCriteria Pydantic model to MongoDB filter dict
            # We need to:
            # 1. Convert to dict
            # 2. Remove None values (optional fields not provided)
            # 3. Use MongoDB field names (camelCase) via aliases
            
            # Get dict with aliases (camelCase field names)
            criteria_dict = criteria.model_dump(by_alias=True, exclude_none=True)
            
            # Call repository with the filter
            sku_docs = self.repository.find_by_criteria(criteria_dict)
            
            # Convert each MongoDB document to a Pydantic SKU model
            skus = [SKU(**doc) for doc in sku_docs]
            
            return skus
            
        except Exception as e:
            raise Exception(f"Error searching SKUs: {str(e)}")
    
    def batch_import(self, skus: List[SKU], validate_only: bool = False) -> BatchImportResult:
        """
        Import multiple SKUs with validation and error handling.
        
        This method performs a batch import operation with the following behavior:
        - All SKUs are validated before any insertion occurs
        - If any SKU fails validation, the entire batch is rejected (all-or-nothing)
        - Duplicate trade numbers are detected and reported
        - Returns detailed results including success/failure counts and error details
        
        Business Logic:
        - Validation happens at two levels:
          1. Pydantic model validation (schema, types, constraints)
          2. Database constraints (unique trade numbers)
        - If validate_only=True, no data is inserted (dry-run mode)
        - Batch operations are atomic: all succeed or all fail
        
        Args:
            skus: List of SKU models to import
            validate_only: If True, only validate without inserting (default: False)
            
        Returns:
            BatchImportResult with counts and error details
            
        Example:
            skus = [
                SKU(trade_number="50624", customer_name="CHICK FIL A INC", ...),
                SKU(trade_number="50625", customer_name="ACME CORP", ...)
            ]
            
            # Validate and import
            result = service.batch_import(skus)
            print(f"Imported {result.successful} of {result.total} SKUs")
            
            # Validate only (dry-run)
            result = service.batch_import(skus, validate_only=True)
            if result.failed == 0:
                print("All SKUs are valid")
        """
        total = len(skus)
        errors = []
        
        try:
            # Step 1: Validate all SKUs
            # Pydantic models are already validated when created, but we check
            # for any additional business rules here
            
            # Check for duplicate trade numbers within the batch
            trade_numbers = [sku.trade_number for sku in skus]
            seen = set()
            for idx, tn in enumerate(trade_numbers):
                if tn in seen:
                    errors.append({
                        "index": idx,
                        "trade_number": tn,
                        "error": f"Duplicate trade number in batch: {tn}"
                    })
                seen.add(tn)
            
            # If validation errors found, return failure result
            if errors:
                return BatchImportResult(
                    total=total,
                    successful=0,
                    failed=len(errors),
                    errors=errors
                )
            
            # If validate_only mode, return success without inserting
            if validate_only:
                return BatchImportResult(
                    total=total,
                    successful=total,
                    failed=0,
                    errors=[]
                )
            
            # Step 2: Convert SKU models to MongoDB documents
            # We need to:
            # 1. Convert to dict with camelCase field names (aliases)
            # 2. Set _id field to trade_number value
            sku_documents = []
            for sku in skus:
                doc = sku.model_dump(by_alias=True)
                # Set _id to tradeNumber for MongoDB primary key
                doc["_id"] = sku.trade_number
                sku_documents.append(doc)
            
            # Step 3: Perform batch insert
            # This is an atomic operation - all succeed or all fail
            inserted_count = self.repository.insert_many(sku_documents)
            
            return BatchImportResult(
                total=total,
                successful=inserted_count,
                failed=0,
                errors=[]
            )
            
        except DuplicateKeyError as e:
            # One or more SKUs already exist in the database
            # Extract the duplicate trade number from the error message if possible
            error_msg = str(e)
            
            # MongoDB DuplicateKeyError includes the duplicate key value
            # Format: "E11000 duplicate key error collection: ... dup key: { _id: \"50624\" }"
            duplicate_tn = "unknown"
            if "_id:" in error_msg:
                # Try to extract the trade number from error message
                try:
                    start = error_msg.index("_id:") + 4
                    end = error_msg.index("}", start)
                    duplicate_tn = error_msg[start:end].strip().strip('"')
                except:
                    pass
            
            return BatchImportResult(
                total=total,
                successful=0,
                failed=total,
                errors=[{
                    "trade_number": duplicate_tn,
                    "error": f"SKU with trade number already exists in database"
                }]
            )
            
        except Exception as e:
            # Unexpected error during batch import
            return BatchImportResult(
                total=total,
                successful=0,
                failed=total,
                errors=[{
                    "error": f"Batch import failed: {str(e)}"
                }]
            )
    
    def export_all(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[SKU]:
        """
        Export all SKUs from the database, optionally filtered.
        
        This method retrieves all SKUs (or a filtered subset) for export purposes.
        Use with caution on large collections as it loads all data into memory.
        
        Business Logic:
        - If no filter provided, returns all SKUs
        - Filter uses MongoDB query syntax (not SearchCriteria model)
        - Results are not paginated (all returned at once)
        - For large datasets, consider adding pagination
        
        Args:
            filter_criteria: Optional MongoDB filter dict (default: None = all SKUs)
            
        Returns:
            List of all SKU models matching the filter
            Returns empty list if no SKUs exist
            
        Raises:
            Exception: For database errors
            
        Example:
            # Export all SKUs
            all_skus = service.export_all()
            print(f"Exported {len(all_skus)} SKUs")
            
            # Export filtered SKUs
            fds_skus = service.export_all({"customerType": "FDS"})
            print(f"Exported {len(fds_skus)} FDS SKUs")
        
        Performance Note:
            For large collections (>10,000 SKUs), consider:
            - Adding pagination
            - Streaming results instead of loading all into memory
            - Using cursor-based iteration
        """
        try:
            if filter_criteria:
                # Use provided filter
                sku_docs = self.repository.find_by_criteria(filter_criteria)
            else:
                # Get all SKUs
                sku_docs = self.repository.find_all()
            
            # Convert documents to Pydantic models
            skus = [SKU(**doc) for doc in sku_docs]
            
            return skus
            
        except Exception as e:
            raise Exception(f"Error exporting SKUs: {str(e)}")

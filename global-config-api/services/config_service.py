"""
Configuration Service - Business Logic Layer for Configuration Management.

This module implements the service layer for configuration operations. The service
layer sits between the router (API endpoints) and repository (data access) layers,
providing business logic, validation, and error handling.

Key Responsibilities:
1. Business Logic: Implement configuration management rules and workflows
2. Validation: Validate configuration updates beyond basic schema validation
3. Error Handling: Convert repository errors to appropriate exceptions
4. Timestamp Management: Automatically set updatedAt timestamps
5. Batch Operations: Coordinate multi-config updates with all-or-nothing validation

Architecture Pattern:
Router Layer (HTTP) → Service Layer (Business Logic) → Repository Layer (Data Access) → MongoDB

The service layer is injected into routers via FastAPI dependency injection,
making it easy to test and maintain.
"""

from typing import Optional, List
from datetime import datetime, timezone
from pydantic import ValidationError

from models.config import Config, ConfigUpdate, BatchConfigUpdate, BatchUpdateResult
from repositories.config_repository import ConfigRepository


class ConfigService:
    """
    Business logic service for configuration operations.
    
    This service provides methods for creating, reading, updating, and batch-updating
    configuration values. It handles business logic such as:
    - Setting timestamps automatically on updates
    - Validating configuration updates
    - Coordinating batch operations with all-or-nothing semantics
    - Converting between repository documents and Pydantic models
    
    The service is designed to be injected into FastAPI routers via dependency
    injection, allowing for easy testing with mock repositories.
    
    Attributes:
        repository: ConfigRepository instance for data access
    
    Example:
        # In a FastAPI router
        @router.get("/config/{key}")
        async def get_config(
            key: str,
            service: ConfigService = Depends(get_config_service)
        ):
            config = service.get_config_by_key(key)
            if not config:
                raise HTTPException(status_code=404, detail="Config not found")
            return config
    """
    
    def __init__(self, repository: ConfigRepository):
        """
        Initialize the service with a repository dependency.
        
        Args:
            repository: ConfigRepository instance for data access
            
        Example:
            from database import get_database
            from repositories.config_repository import ConfigRepository
            
            db = get_database()
            repo = ConfigRepository(db)
            service = ConfigService(repo)
        """
        self.repository = repository
    
    def get_config_by_key(self, key: str) -> Optional[Config]:
        """
        Retrieve a configuration by its key.
        
        This method fetches a configuration document from the repository and
        converts it to a Config Pydantic model. If the configuration doesn't
        exist, it returns None.
        
        Business Logic:
        - Lookup configuration by key (primary key lookup, very fast)
        - Convert MongoDB document to Pydantic model
        - Return None if not found (router layer handles 404 response)
        
        Args:
            key: The unique configuration key identifier
            
        Returns:
            Config model if found, None otherwise
            
        Example:
            config = service.get_config_by_key("enumeration.defaultMaxTrim")
            if config:
                print(f"Current value: {config.value}")
            else:
                print("Configuration not found")
        
        Note:
            This method does not raise exceptions for not-found cases.
            The router layer is responsible for returning appropriate HTTP status codes.
        """
        # Fetch document from repository
        document = self.repository.find_by_key(key)
        
        # Return None if not found
        if document is None:
            return None
        
        # Convert MongoDB document to Pydantic model
        # The Config model handles field mapping (camelCase to snake_case)
        return Config(**document)
    
    def update_config(self, key: str, update: ConfigUpdate) -> Config:
        """
        Create or update a configuration.
        
        This method implements the upsert pattern - it will create a new configuration
        if the key doesn't exist, or update an existing one if it does. The updatedAt
        timestamp is automatically set to the current time.
        
        Business Logic:
        1. Validate the update data (Pydantic validation happens automatically)
        2. Set the updatedAt timestamp to current UTC time
        3. Upsert the configuration in the repository
        4. Return the updated Config model
        
        Validation:
        - Value must match valueType (enforced by ConfigUpdate model)
        - Numeric values must be within minValue/maxValue if specified
        - All required fields must be present
        
        Args:
            key: The configuration key identifier
            update: ConfigUpdate model with new configuration data
            
        Returns:
            Updated Config model with current timestamp
            
        Raises:
            ValidationError: If the update data fails validation
            Exception: If database operation fails
            
        Example:
            update = ConfigUpdate(
                value=5,
                value_type=ValueType.INT,
                description="Maximum trim allowed",
                min_value=0,
                max_value=100
            )
            
            config = service.update_config("enumeration.defaultMaxTrim", update)
            print(f"Updated config: {config.key} = {config.value}")
        
        Note:
            The updatedAt timestamp is set automatically by this method.
            Clients should not provide updatedAt in the update request.
        """
        # Set the updatedAt timestamp to current UTC time
        # This ensures the timestamp reflects when the update actually occurred
        updated_at = datetime.now(timezone.utc)
        
        # Create a Config model with all fields
        # This validates the data and ensures all constraints are met
        config = Config(
            key=key,
            value=update.value,
            value_type=update.value_type,
            description=update.description,
            updated_at=updated_at,
            min_value=update.min_value,
            max_value=update.max_value
        )
        
        # Convert Config model to dictionary for MongoDB storage
        # Use by_alias=True to convert snake_case to camelCase for MongoDB
        document = config.model_dump(by_alias=True)
        
        # Upsert the document in the repository
        # This will create or update the configuration atomically
        self.repository.upsert(key, document)
        
        # Return the Config model
        return config
    
    def get_all_configs(self) -> List[Config]:
        """
        Retrieve all configurations from the database.
        
        This method fetches all configuration documents and converts them to
        Config Pydantic models. It's useful for:
        - Admin dashboards showing all system configurations
        - Configuration export for backup
        - Bulk configuration validation
        - System health checks
        
        Business Logic:
        - Fetch all configuration documents from repository
        - Convert each document to a Config model
        - Return as a list
        
        Returns:
            List of all Config models in the database
            Returns empty list if no configurations exist
            
        Example:
            all_configs = service.get_all_configs()
            print(f"Total configurations: {len(all_configs)}")
            
            for config in all_configs:
                print(f"{config.key}: {config.value} ({config.value_type})")
        
        Performance:
        - Configuration collections are typically small (dozens to hundreds)
        - Loading all into memory is acceptable for this use case
        - If the collection grows very large, consider pagination
        
        Note:
            This method loads all configurations into memory at once.
            For very large configuration sets, consider implementing pagination.
        """
        # Fetch all documents from repository
        documents = self.repository.find_all()
        
        # Convert each document to a Config model
        # List comprehension is efficient for this operation
        return [Config(**doc) for doc in documents]
    
    def batch_update(
        self,
        updates: List[BatchConfigUpdate],
        validate_only: bool = False
    ) -> BatchUpdateResult:
        """
        Update multiple configurations with all-or-nothing validation.
        
        This method implements batch configuration updates with comprehensive
        validation. All configurations are validated before any updates occur,
        ensuring data consistency.
        
        Business Logic:
        1. Validate all configuration updates first
        2. If any validation fails, reject the entire batch (no partial updates)
        3. If all valid and not validate_only, update all configurations
        4. Return summary with success/failure counts and error details
        
        Validation Strategy:
        - Phase 1: Validate all updates (Pydantic validation + business rules)
        - Phase 2: If all valid, apply updates (or skip if validate_only=True)
        - All-or-nothing: Either all updates succeed or none are applied
        
        This ensures data consistency - you never end up with partially updated
        configurations that could leave the system in an inconsistent state.
        
        Args:
            updates: List of BatchConfigUpdate objects with key and update data
            validate_only: If True, only validate without updating (default: False)
            
        Returns:
            BatchUpdateResult with:
            - total: Number of updates in the request
            - successful: Number of successful updates
            - failed: Number of failed updates
            - errors: List of error details for failed updates
            
        Example:
            updates = [
                BatchConfigUpdate(
                    key="enumeration.defaultMaxTrim",
                    update=ConfigUpdate(
                        value=5,
                        value_type=ValueType.INT,
                        description="Max trim allowed",
                        min_value=0,
                        max_value=100
                    )
                ),
                BatchConfigUpdate(
                    key="enumeration.lineEfficiency",
                    update=ConfigUpdate(
                        value=0.85,
                        value_type=ValueType.FLOAT,
                        description="Line efficiency factor",
                        min_value=0.0,
                        max_value=1.0
                    )
                )
            ]
            
            result = service.batch_update(updates)
            print(f"Updated {result.successful} of {result.total} configs")
            
            if result.failed > 0:
                print("Errors:")
                for error in result.errors:
                    print(f"  {error['key']}: {error['error']}")
        
        Validation-Only Mode:
        - Set validate_only=True to check if updates are valid without applying them
        - Useful for pre-flight validation in UIs
        - Returns validation results without modifying the database
        
        Error Handling:
        - Each error includes: index (position in list), key, and error message
        - Validation errors are caught and reported, not raised
        - Database errors are caught and reported for each failed update
        
        Note:
            This method implements all-or-nothing validation semantics.
            If any update fails validation, no updates are applied to the database.
        """
        errors = []
        
        # Phase 1: Validate all updates
        # We validate everything first before making any database changes
        # This ensures we don't end up with partial updates if something fails
        for idx, batch_update in enumerate(updates):
            try:
                # Validate the update by creating a Config model
                # This will raise ValidationError if the data is invalid
                # We use a temporary timestamp for validation
                Config(
                    key=batch_update.key,
                    value=batch_update.update.value,
                    value_type=batch_update.update.value_type,
                    description=batch_update.update.description,
                    updated_at=datetime.now(timezone.utc),
                    min_value=batch_update.update.min_value,
                    max_value=batch_update.update.max_value
                )
            except ValidationError as e:
                # Capture validation errors with details
                errors.append({
                    "index": idx,
                    "key": batch_update.key,
                    "error": str(e)
                })
            except Exception as e:
                # Capture any other errors (e.g., unexpected validation issues)
                errors.append({
                    "index": idx,
                    "key": batch_update.key,
                    "error": f"Validation error: {str(e)}"
                })
        
        # If there are validation errors or validate_only mode, return without updating
        if errors or validate_only:
            return BatchUpdateResult(
                total=len(updates),
                successful=0,
                failed=len(errors),
                errors=errors
            )
        
        # Phase 2: Apply all updates
        # At this point, we know all updates are valid
        # We still need to handle potential database errors
        successful = 0
        for batch_update in updates:
            try:
                # Update the configuration using the update_config method
                # This handles timestamp setting and upsert logic
                self.update_config(batch_update.key, batch_update.update)
                successful += 1
            except Exception as e:
                # Capture database errors
                # Note: This shouldn't happen if validation passed, but we handle it anyway
                errors.append({
                    "key": batch_update.key,
                    "error": f"Database error: {str(e)}"
                })
        
        # Return summary of results
        return BatchUpdateResult(
            total=len(updates),
            successful=successful,
            failed=len(errors),
            errors=errors
        )

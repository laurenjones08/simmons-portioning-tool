"""
Configuration Repository - Data Access Layer for Global Config Collection.

This module implements the repository pattern for configuration data access. The repository
pattern provides an abstraction layer between the business logic (service layer)
and data storage (MongoDB), making the code more maintainable and testable.

Key Benefits of Repository Pattern:
1. Separation of Concerns: Business logic doesn't need to know about MongoDB queries
2. Testability: Easy to mock the repository in service layer tests
3. Maintainability: Database query logic is centralized in one place
4. Flexibility: Can swap out MongoDB for another database with minimal changes

MongoDB Query Patterns Used:
- find_one(): Retrieve a single document by _id (O(1) with index)
- update_one() with upsert: Create or update a document in a single operation
- find(): Retrieve multiple documents (all configs in this case)
- Indexes: _id is automatically indexed for fast lookups
"""

from typing import List, Optional, Dict, Any
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import PyMongoError


class ConfigRepository:
    """
    Data access layer for the global configuration collection.
    
    This repository provides methods to interact with the MongoDB 'global_config' collection.
    All methods handle MongoDB-specific operations and return plain dictionaries
    that can be converted to Pydantic models by the service layer.
    
    Collection Schema:
    - Collection name: "global_config"
    - Primary key: _id (set to configuration key value)
    - Document structure: See Config model in models/config.py
    
    Attributes:
        collection: PyMongo Collection object for the 'global_config' collection
    """
    
    def __init__(self, database: Database):
        """
        Initialize the repository with a MongoDB database connection.
        
        Args:
            database: PyMongo Database instance (injected via FastAPI dependency)
            
        Example:
            from database import get_database
            
            db = get_database()
            repo = ConfigRepository(db)
        """
        # Get the 'global_config' collection from the database
        # If the collection doesn't exist, MongoDB will create it on first insert
        self.collection: Collection = database["global_config"]
    
    def find_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Find a single configuration document by its key.
        
        This method performs a primary key lookup using the _id field, which is
        automatically indexed by MongoDB. This makes the query very fast (O(1)).
        
        MongoDB Query Pattern:
        - find_one({"_id": value}): Returns the first document matching the filter
        - Returns None if no document is found
        - The _id field is unique, so at most one document will match
        
        Args:
            key: The unique configuration key identifier (stored as _id)
            
        Returns:
            Dictionary containing the configuration document if found, None otherwise
            
        Example:
            repo = ConfigRepository(db)
            config = repo.find_by_key("enumeration.defaultMaxTrim")
            
            if config:
                print(f"Config value: {config['value']}")
            else:
                print("Configuration not found")
        
        Note:
            The returned dictionary uses camelCase field names as stored in MongoDB.
            The service layer is responsible for converting to Pydantic models.
        """
        try:
            # find_one() returns None if no document matches
            # Using _id for lookup is the fastest query in MongoDB
            return self.collection.find_one({"_id": key})
        except PyMongoError as e:
            # Log the error and re-raise for the service layer to handle
            # In production, you might want to use proper logging here
            raise Exception(f"Database error finding config by key: {str(e)}")
    
    def upsert(self, key: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Insert or update a configuration document (upsert operation).
        
        This method implements the "upsert" pattern - a combination of update and insert.
        If a document with the specified key exists, it will be replaced with the new
        document. If no document exists, a new one will be created.
        
        MongoDB Upsert Pattern:
        - update_one(filter, {"$set": document}, upsert=True)
        - upsert=True: Create document if it doesn't exist
        - $set operator: Updates specified fields (or sets them if new document)
        - Returns UpdateResult with matched_count, modified_count, upserted_id
        
        Why Upsert is Useful:
        - Eliminates the need for separate "check if exists" logic
        - Atomic operation: No race conditions between check and insert/update
        - Simplifies API: Single endpoint can handle both create and update
        - Idempotent: Calling multiple times with same data has same effect
        
        Alternative Approach (replace_one):
        - replace_one() replaces the entire document (not just specified fields)
        - We use replace_one here because we want to replace the whole config
        
        Args:
            key: The configuration key identifier (used as _id)
            document: Dictionary containing the complete configuration data
                     Must include all required fields for the Config model
            
        Returns:
            The upserted document (same as input with _id set)
            
        Example:
            config_doc = {
                "_id": "enumeration.defaultMaxTrim",
                "key": "enumeration.defaultMaxTrim",
                "value": 2,
                "valueType": "int",
                "description": "Default max trim allowed",
                "updatedAt": "2024-03-08T10:30:00Z",
                "minValue": 0,
                "maxValue": 100
            }
            
            # This will create or update the config
            result = repo.upsert("enumeration.defaultMaxTrim", config_doc)
            print(f"Upserted config: {result['key']}")
        
        Performance:
        - Very fast due to _id index lookup
        - Single round-trip to database
        - Atomic operation (no race conditions)
        
        Note:
            The document should have _id field set to the key value before calling
            this method. The service layer is responsible for setting this field.
        """
        try:
            # Ensure the _id field is set to the key
            # This makes the key the primary identifier in MongoDB
            document["_id"] = key
            
            # replace_one() with upsert=True implements the upsert pattern
            # - If document with _id=key exists: replace it entirely
            # - If no document exists: insert the new document
            # This is atomic - no race conditions
            result = self.collection.replace_one(
                {"_id": key},  # Filter: find document with this _id
                document,       # Replacement: the new document
                upsert=True     # Create if doesn't exist
            )
            
            # Return the document that was upserted
            # We return the input document since we know it was successfully saved
            return document
            
        except PyMongoError as e:
            raise Exception(f"Database error upserting config: {str(e)}")
    
    def find_all(self) -> List[Dict[str, Any]]:
        """
        Retrieve all configuration documents from the collection.
        
        This method returns every document in the global_config collection.
        Since configuration collections are typically small (dozens to hundreds
        of configs, not thousands), loading all into memory is acceptable.
        
        MongoDB Query Pattern:
        - find() with no filter: Returns cursor over all documents
        - Equivalent to find({})
        - list(cursor): Converts cursor to list of documents
        
        Performance Considerations:
        - Config collections are typically small, so this is efficient
        - All documents are loaded into memory at once
        - If config collection grows very large (>1000 docs), consider pagination
        
        Args:
            None
            
        Returns:
            List of all configuration document dictionaries in the collection
            Returns empty list if collection is empty
            
        Example:
            # Get all configurations
            all_configs = repo.find_all()
            print(f"Total configs: {len(all_configs)}")
            
            # Display each config
            for config in all_configs:
                print(f"{config['key']}: {config['value']}")
        
        Use Cases:
        - Admin dashboard showing all system configurations
        - Configuration export for backup
        - Bulk configuration validation
        - System health checks
        
        Note:
            Returned documents use camelCase field names as stored in MongoDB.
            The service layer converts these to Pydantic models.
        """
        try:
            # find() with no arguments returns all documents
            # This is equivalent to find({})
            cursor = self.collection.find()
            
            # Convert cursor to list to fetch all documents
            # For small collections this is fine; for large ones consider pagination
            return list(cursor)
            
        except PyMongoError as e:
            raise Exception(f"Database error retrieving all configs: {str(e)}")
    
    def delete_by_key(self, key: str) -> bool:
        """
        Delete a single configuration document by its key.
        
        This method removes a configuration from the collection. It's included
        for completeness but may not be required by the current requirements.
        
        MongoDB Delete Pattern:
        - delete_one(filter): Deletes the first document matching the filter
        - Returns DeleteResult with deleted_count
        
        Args:
            key: The unique configuration key identifier
            
        Returns:
            True if a document was deleted, False if no document was found
            
        Example:
            success = repo.delete_by_key("enumeration.defaultMaxTrim")
            if success:
                print("Config deleted")
            else:
                print("Config not found")
        
        Note:
            Deleting configurations should be done carefully as it may affect
            system behavior. Consider soft deletes or archiving instead.
        """
        try:
            result = self.collection.delete_one({"_id": key})
            return result.deleted_count > 0
        except PyMongoError as e:
            raise Exception(f"Database error deleting config: {str(e)}")

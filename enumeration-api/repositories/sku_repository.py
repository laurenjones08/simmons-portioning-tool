"""
SKU Repository - Data Access Layer for SKU Collection.

This module implements the repository pattern for SKU data access. The repository
pattern provides an abstraction layer between the business logic (service layer)
and data storage (MongoDB), making the code more maintainable and testable.

Key Benefits of Repository Pattern:
1. Separation of Concerns: Business logic doesn't need to know about MongoDB queries
2. Testability: Easy to mock the repository in service layer tests
3. Maintainability: Database query logic is centralized in one place
4. Flexibility: Can swap out MongoDB for another database with minimal changes

MongoDB Query Patterns Used:
- find_one(): Retrieve a single document by _id (O(1) with index)
- find(): Retrieve multiple documents matching a filter (uses indexes when possible)
- insert_many(): Bulk insert multiple documents in a single operation
- Indexes: _id is automatically indexed; additional indexes improve search performance
"""

from typing import List, Optional, Dict, Any
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import PyMongoError, DuplicateKeyError


class SKURepository:
    """
    Data access layer for the SKU collection.
    
    This repository provides methods to interact with the MongoDB 'skus' collection.
    All methods handle MongoDB-specific operations and return plain dictionaries
    that can be converted to Pydantic models by the service layer.
    
    Collection Schema:
    - Collection name: "skus"
    - Primary key: _id (set to tradeNumber value)
    - Document structure: See SKU model in models/sku.py
    
    Attributes:
        collection: PyMongo Collection object for the 'skus' collection
    """
    
    def __init__(self, database: Database):
        """
        Initialize the repository with a MongoDB database connection.
        
        Args:
            database: PyMongo Database instance (injected via FastAPI dependency)
            
        Example:
            from database import get_database
            
            db = get_database()
            repo = SKURepository(db)
        """
        # Get the 'skus' collection from the database
        # If the collection doesn't exist, MongoDB will create it on first insert
        self.collection: Collection = database["skus"]
    
    def find_by_trade_number(self, trade_number: str) -> Optional[Dict[str, Any]]:
        """
        Find a single SKU document by its trade number.
        
        This method performs a primary key lookup using the _id field, which is
        automatically indexed by MongoDB. This makes the query very fast (O(1)).
        
        MongoDB Query Pattern:
        - find_one({"_id": value}): Returns the first document matching the filter
        - Returns None if no document is found
        - The _id field is unique, so at most one document will match
        
        Args:
            trade_number: The unique trade number identifier (stored as _id)
            
        Returns:
            Dictionary containing the SKU document if found, None otherwise
            
        Example:
            repo = SKURepository(db)
            sku = repo.find_by_trade_number("50624")
            
            if sku:
                print(f"Found SKU: {sku['customerName']}")
            else:
                print("SKU not found")
        
        Note:
            The returned dictionary uses camelCase field names as stored in MongoDB.
            The service layer is responsible for converting to Pydantic models.
        """
        try:
            # find_one() returns None if no document matches
            # Using _id for lookup is the fastest query in MongoDB
            return self.collection.find_one({"_id": trade_number})
        except PyMongoError as e:
            # Log the error and re-raise for the service layer to handle
            # In production, you might want to use proper logging here
            raise Exception(f"Database error finding SKU by trade number: {str(e)}")
    
    def find_by_criteria(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find SKU documents matching the specified filter criteria.
        
        This method performs a filtered search on the SKU collection. Multiple
        criteria are combined with AND logic (all must match). MongoDB will use
        indexes when available to optimize the query.
        
        MongoDB Query Pattern:
        - find(filter): Returns a cursor over all matching documents
        - Empty filter {} matches all documents
        - Multiple fields in filter are combined with AND logic
        - list(cursor): Converts cursor to a list of documents
        
        Performance Considerations:
        - Queries on indexed fields (customerType, productType, prodPlant) are fast
        - Queries on non-indexed fields require collection scans (slower)
        - Consider adding indexes for frequently searched fields
        
        Args:
            criteria: Dictionary of field filters (e.g., {"customerType": "FDS"})
                     Empty dict {} returns all documents
            
        Returns:
            List of dictionaries, each containing a matching SKU document
            Returns empty list if no documents match
            
        Example:
            # Search by single criterion
            skus = repo.find_by_criteria({"customerType": "FDS"})
            
            # Search by multiple criteria (AND logic)
            skus = repo.find_by_criteria({
                "customerType": "FDS",
                "productType": "NUGGET",
                "prodPlant": "FSP"
            })
            
            # Get all SKUs
            all_skus = repo.find_by_criteria({})
        
        Note:
            - Returned documents use camelCase field names
            - The service layer converts these to Pydantic models
            - For large result sets, consider adding pagination
        """
        try:
            # find() returns a cursor, which we convert to a list
            # The cursor is lazy - it only fetches documents as needed
            # Converting to list fetches all documents into memory
            cursor = self.collection.find(criteria)
            return list(cursor)
        except PyMongoError as e:
            raise Exception(f"Database error searching SKUs: {str(e)}")
    
    def insert(self, sku_document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Insert a single SKU document into the collection.
        
        This method inserts a new SKU document. The _id field must be set to the
        tradeNumber value before calling this method. If a document with the same
        _id already exists, a DuplicateKeyError will be raised.
        
        MongoDB Insert Pattern:
        - insert_one(document): Inserts a single document
        - Returns InsertOneResult with inserted_id
        - Raises DuplicateKeyError if _id already exists
        
        Args:
            sku_document: Dictionary containing the SKU data with _id field set
            
        Returns:
            The inserted document (same as input)
            
        Raises:
            DuplicateKeyError: If a SKU with the same trade number already exists
            Exception: For other database errors
            
        Example:
            sku_doc = {
                "_id": "50624",
                "tradeNumber": "50624",
                "customerName": "CHICK FIL A INC",
                # ... other fields
            }
            
            try:
                result = repo.insert(sku_doc)
                print(f"Inserted SKU: {result['_id']}")
            except DuplicateKeyError:
                print("SKU already exists")
        """
        try:
            # insert_one() adds the document to the collection
            # The _id field must be unique
            result = self.collection.insert_one(sku_document)
            
            # Return the document with its _id
            # In this case, _id was already set, so we just return the input
            return sku_document
        except DuplicateKeyError:
            # Re-raise DuplicateKeyError so service layer can handle it
            raise
        except PyMongoError as e:
            raise Exception(f"Database error inserting SKU: {str(e)}")
    
    def update(self, trade_number: str, sku_document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update an existing SKU document.

        Args:
            trade_number: The trade number (_id) of the SKU to update
            sku_document: Dictionary containing the updated SKU data

        Returns:
            The updated document if found, None otherwise
        """
        try:
            result = self.collection.replace_one(
                {"_id": trade_number},
                sku_document
            )
            if result.matched_count > 0:
                return sku_document
            return None
        except PyMongoError as e:
            raise Exception(f"Database error updating SKU: {str(e)}")

    def insert_many(self, sku_documents: List[Dict[str, Any]]) -> int:
        """
        Insert multiple SKU documents in a single batch operation.
        
        This method performs a bulk insert, which is much more efficient than
        inserting documents one at a time. All documents are inserted in a single
        round-trip to the database.
        
        MongoDB Bulk Insert Pattern:
        - insert_many(documents, ordered=True): Inserts multiple documents
        - ordered=True: Stops on first error (default behavior)
        - ordered=False: Continues inserting even if some fail
        - Returns InsertManyResult with inserted_ids list
        
        Atomicity:
        - With ordered=True, if any insert fails, remaining documents are not inserted
        - This provides "all or nothing" semantics for the batch
        - MongoDB does not support multi-document transactions in all configurations
        
        Args:
            sku_documents: List of SKU document dictionaries, each with _id field set
            
        Returns:
            Number of documents successfully inserted
            
        Raises:
            DuplicateKeyError: If any SKU trade number already exists (with ordered=True)
            Exception: For other database errors
            
        Example:
            sku_docs = [
                {"_id": "50624", "tradeNumber": "50624", ...},
                {"_id": "50625", "tradeNumber": "50625", ...},
                {"_id": "50626", "tradeNumber": "50626", ...}
            ]
            
            try:
                count = repo.insert_many(sku_docs)
                print(f"Inserted {count} SKUs")
            except DuplicateKeyError as e:
                print(f"Duplicate SKU found: {e}")
        
        Performance:
        - Much faster than individual inserts (single network round-trip)
        - For very large batches (>1000 docs), consider chunking
        - MongoDB has a 16MB document size limit per operation
        """
        try:
            if not sku_documents:
                return 0
            
            # insert_many() performs bulk insert
            # ordered=True means stop on first error (all-or-nothing behavior)
            result = self.collection.insert_many(sku_documents, ordered=True)
            
            # Return the count of inserted documents
            return len(result.inserted_ids)
        except DuplicateKeyError:
            # Re-raise so service layer can provide user-friendly error
            raise
        except PyMongoError as e:
            raise Exception(f"Database error during bulk insert: {str(e)}")
    
    def find_all(self) -> List[Dict[str, Any]]:
        """
        Retrieve all SKU documents from the collection.
        
        This method returns every document in the SKUs collection. Use with caution
        on large collections, as it loads all documents into memory.
        
        MongoDB Query Pattern:
        - find() with no filter: Returns cursor over all documents
        - Equivalent to find({})
        
        Performance Considerations:
        - For large collections (>10,000 docs), this can be slow and memory-intensive
        - Consider adding pagination for production use
        - Consider adding filters to limit the result set
        
        Args:
            None
            
        Returns:
            List of all SKU document dictionaries in the collection
            Returns empty list if collection is empty
            
        Example:
            # Export all SKUs
            all_skus = repo.find_all()
            print(f"Total SKUs: {len(all_skus)}")
            
            # Process each SKU
            for sku in all_skus:
                print(f"Trade Number: {sku['tradeNumber']}")
        
        Alternative for Large Collections:
            # Use pagination instead
            page_size = 100
            skip = 0
            while True:
                batch = list(collection.find().skip(skip).limit(page_size))
                if not batch:
                    break
                # Process batch
                skip += page_size
        """
        try:
            # find() with no arguments returns all documents
            # This is equivalent to find({})
            cursor = self.collection.find()
            return list(cursor)
        except PyMongoError as e:
            raise Exception(f"Database error retrieving all SKUs: {str(e)}")
    
    def delete_by_trade_number(self, trade_number: str) -> bool:
        """
        Delete a single SKU document by its trade number.
        
        This method removes a SKU from the collection. It's included for completeness
        but may not be required by the current requirements.
        
        MongoDB Delete Pattern:
        - delete_one(filter): Deletes the first document matching the filter
        - Returns DeleteResult with deleted_count
        
        Args:
            trade_number: The unique trade number identifier
            
        Returns:
            True if a document was deleted, False if no document was found
            
        Example:
            success = repo.delete_by_trade_number("50624")
            if success:
                print("SKU deleted")
            else:
                print("SKU not found")
        """
        try:
            result = self.collection.delete_one({"_id": trade_number})
            return result.deleted_count > 0
        except PyMongoError as e:
            raise Exception(f"Database error deleting SKU: {str(e)}")

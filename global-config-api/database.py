"""
Database connection module for the Global Config API.

This module manages MongoDB connections using PyMongo and provides dependency injection
for FastAPI routes. It follows the FastAPI dependency injection pattern to ensure:
1. Database connections are properly managed
2. Each request gets access to the database
3. Resources are cleaned up automatically
4. Testing can easily mock database dependencies

Key Concepts:
- PyMongo MongoClient: Thread-safe MongoDB client with connection pooling
- FastAPI Depends(): Dependency injection system for routes
- Generator pattern: Used to provide and cleanup resources

Usage in a FastAPI route:
    from fastapi import Depends
    from database import get_database
    
    @app.get("/config/{key}")
    async def get_config(
        key: str,
        db: Database = Depends(get_database)
    ):
        # db is automatically injected by FastAPI
        config = db["global_config"].find_one({"_id": key})
        return config
"""

from typing import Generator
from pymongo import MongoClient
from pymongo.database import Database
from config import get_settings


# Global MongoDB client instance
# MongoClient is thread-safe and maintains an internal connection pool
# It's recommended to create a single client instance and reuse it
# The client automatically handles connection pooling and reconnection
_mongo_client: MongoClient = None


def get_mongo_client() -> MongoClient:
    """
    Get or create the MongoDB client instance.
    
    This function implements a singleton pattern for the MongoClient.
    The client is created once and reused for all database operations.
    
    MongoClient features:
    - Thread-safe: Can be used across multiple threads
    - Connection pooling: Maintains a pool of connections for efficiency
    - Auto-reconnect: Automatically reconnects if connection is lost
    
    Returns:
        MongoClient: Singleton MongoDB client instance
        
    Example:
        client = get_mongo_client()
        db = client["my_database"]
        collection = db["my_collection"]
    """
    global _mongo_client
    
    if _mongo_client is None:
        settings = get_settings()
        
        # Create MongoClient with connection string from settings
        # The client will parse the URL and establish connections as needed
        _mongo_client = MongoClient(
            settings.mongodb_url,
            # serverSelectionTimeoutMS: How long to wait for server selection
            serverSelectionTimeoutMS=5000,  # 5 seconds
            # connectTimeoutMS: How long to wait for a connection to be established
            connectTimeoutMS=5000,  # 5 seconds
        )
        
    return _mongo_client


def get_database() -> Generator[Database, None, None]:
    """
    FastAPI dependency that provides MongoDB database instance.
    
    This is a generator function that yields a Database instance for use in
    FastAPI route handlers. The generator pattern allows FastAPI to:
    1. Execute code before the request (setup)
    2. Yield the dependency to the route handler
    3. Execute code after the request (cleanup)
    
    The Database object provides access to MongoDB collections:
    - db["collection_name"] returns a Collection object
    - Collections support find(), insert_one(), update_one(), etc.
    
    Yields:
        Database: MongoDB database instance for the configured database
        
    Example usage in a FastAPI route:
        from fastapi import APIRouter, Depends
        from database import get_database
        
        router = APIRouter()
        
        @router.get("/config/{key}")
        async def get_config(
            key: str,
            db: Database = Depends(get_database)
        ):
            # FastAPI automatically calls get_database() and injects db
            config = db["global_config"].find_one({"_id": key})
            if config is None:
                raise HTTPException(status_code=404, detail="Config not found")
            return config
    
    Dependency Injection Benefits:
    - Separation of concerns: Routes don't manage database connections
    - Testability: Easy to mock the database in tests
    - Resource management: Connections are properly handled
    - Type safety: IDE can autocomplete Database methods
    """
    settings = get_settings()
    client = get_mongo_client()
    
    # Get the database instance from the client
    # This doesn't create a new connection, just returns a Database object
    # that references the specified database name
    database = client[settings.mongodb_database]
    
    try:
        # Yield the database to the route handler
        # Everything after 'yield' runs after the request completes
        yield database
    finally:
        # Cleanup code would go here if needed
        # For MongoClient, we don't close connections per-request
        # because the client maintains a connection pool
        # The client will be closed when the application shuts down
        pass


def close_mongo_connection():
    """
    Close the MongoDB client connection.
    
    This function should be called when the application shuts down to properly
    close all connections in the connection pool and release resources.
    
    In FastAPI, this is typically called in a shutdown event handler:
    
        from fastapi import FastAPI
        from database import close_mongo_connection
        
        app = FastAPI()
        
        @app.on_event("shutdown")
        async def shutdown_event():
            close_mongo_connection()
    
    This ensures graceful shutdown and prevents connection leaks.
    """
    global _mongo_client
    
    if _mongo_client is not None:
        # Close all connections in the pool
        _mongo_client.close()
        _mongo_client = None

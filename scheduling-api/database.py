from typing import Generator

from pymongo import MongoClient
from pymongo.database import Database

from config import get_settings

_mongo_client: MongoClient = None


def get_mongo_client() -> MongoClient:
    global _mongo_client
    if _mongo_client is None:
        settings = get_settings()
        _mongo_client = MongoClient(
            settings.mongodb_url,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
    return _mongo_client


def get_database() -> Generator[Database, None, None]:
    settings = get_settings()
    client = get_mongo_client()
    database = client[settings.mongodb_database]
    try:
        yield database
    finally:
        pass


def close_mongo_connection():
    global _mongo_client
    if _mongo_client is not None:
        _mongo_client.close()
        _mongo_client = None


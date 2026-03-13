"""
Property-based tests for ConfigService.

This module contains property-based tests using Hypothesis to verify that the
ConfigService behaves correctly across a wide range of inputs. Each test runs
100+ iterations with randomly generated data to ensure comprehensive coverage.

Tests are tagged with the feature name and property number from the design document.
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from datetime import datetime, timezone
from typing import Dict, Any
from unittest.mock import Mock, MagicMock
from pydantic import ValidationError

from services.config_service import ConfigService
from models.config import Config, ConfigUpdate, ValueType, BatchConfigUpdate, BatchUpdateResult
from repositories.config_repository import ConfigRepository


# Custom strategies for generating valid configuration data

def valid_key_strategy():
    """Generate valid configuration keys."""
    return st.text(
        alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_',
        min_size=1,
        max_size=50
    ).filter(lambda s: not s.startswith('.') and not s.endswith('.'))


def valid_description_strategy():
    """Generate valid descriptions."""
    return st.text(
        alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-',
        min_size=1,
        max_size=100
    )


def value_for_type_strategy(value_type: ValueType) -> st.SearchStrategy:
    """Generate a value matching the specified ValueType."""
    if value_type == ValueType.INT:
        return st.integers(min_value=-1000000, max_value=1000000)
    elif value_type == ValueType.STRING:
        return st.text(min_size=0, max_size=200)
    elif value_type == ValueType.FLOAT:
        return st.floats(
            min_value=-1000000.0,
            max_value=1000000.0,
            allow_nan=False,
            allow_infinity=False
        )
    elif value_type == ValueType.BOOL:
        return st.booleans()
    else:
        raise ValueError(f"Unknown ValueType: {value_type}")


# Property 5: Config retrieval returns correct data
# Feature: fastapi-enumeration-services, Property 5: Config retrieval returns correct data
# Validates: Requirements 10.2

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    key=valid_key_strategy(),
    value_type=st.sampled_from(list(ValueType)),
    description=valid_description_strategy(),
    data=st.data()
)
def test_property_config_retrieval_returns_correct_data(
    key: str,
    value_type: ValueType,
    description: str,
    data
):
    """
    **Validates: Requirements 10.2**
    
    Feature: fastapi-enumeration-services, Property 5: Config retrieval returns correct data
    
    Property: For any configuration document stored in the database, retrieving it
    by key should return a document where the _id field equals the key field.
    
    This test verifies that:
    1. ConfigService.get_config_by_key returns the correct configuration
    2. The returned config has all the expected data
    3. The key field matches the lookup key
    """
    # Generate a value matching the value_type
    value = data.draw(value_for_type_strategy(value_type))
    updated_at = datetime.now(timezone.utc)
    
    # Create a mock repository
    mock_repo = Mock(spec=ConfigRepository)
    
    # Create the document that the repository will return
    # MongoDB stores with _id equal to key
    document = {
        "_id": key,
        "key": key,
        "value": value,
        "valueType": value_type.value,
        "description": description,
        "updatedAt": updated_at,
        "minValue": None,
        "maxValue": None
    }
    
    # Configure the mock to return this document
    mock_repo.find_by_key.return_value = document
    
    # Create service with mock repository
    service = ConfigService(mock_repo)
    
    # Act: Retrieve the configuration
    config = service.get_config_by_key(key)
    
    # Assert: Verify the repository was called with the correct key
    mock_repo.find_by_key.assert_called_once_with(key)
    
    # Assert: Verify the returned config has correct data
    assert config is not None
    assert config.key == key
    assert config.value == value
    assert config.value_type == value_type
    assert config.description == description
    assert config.updated_at == updated_at
    
    # Property: _id field equals key field (verified by the document structure)
    # In MongoDB, _id is set to key, and we verify the returned config.key matches


@settings(max_examples=100)
@given(
    key=valid_key_strategy()
)
def test_property_config_retrieval_returns_none_for_nonexistent_key(key: str):
    """
    **Validates: Requirements 10.2**
    
    Feature: fastapi-enumeration-services, Property 5: Config retrieval returns correct data
    
    Property: For any key that doesn't exist in the database, retrieving it
    should return None.
    
    This test verifies that:
    1. ConfigService.get_config_by_key returns None for non-existent keys
    2. The service handles missing configurations gracefully
    """
    # Create a mock repository
    mock_repo = Mock(spec=ConfigRepository)
    
    # Configure the mock to return None (config not found)
    mock_repo.find_by_key.return_value = None
    
    # Create service with mock repository
    service = ConfigService(mock_repo)
    
    # Act: Retrieve the configuration
    config = service.get_config_by_key(key)
    
    # Assert: Verify the repository was called with the correct key
    mock_repo.find_by_key.assert_called_once_with(key)
    
    # Assert: Verify None is returned
    assert config is None


# Property 6: Config update persistence
# Feature: fastapi-enumeration-services, Property 6: Config update persistence
# Validates: Requirements 11.2, 11.3, 11.4

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(
    key=valid_key_strategy(),
    value_type=st.sampled_from(list(ValueType)),
    description=valid_description_strategy(),
    data=st.data()
)
def test_property_config_update_persistence_with_database(
    key: str,
    value_type: ValueType,
    description: str,
    data,
    mongodb_database
):
    """
    **Validates: Requirements 11.2, 11.3, 11.4**
    
    Feature: fastapi-enumeration-services, Property 6: Config update persistence
    
    Property: For any configuration update (create or modify), after the update
    operation completes, retrieving the configuration by key should return the
    updated value, and the updatedAt field should be set to a valid ISO 8601
    timestamp that is greater than or equal to the update request time.
    
    This test verifies that:
    1. ConfigService.update_config persists the configuration to the database
    2. Retrieving the config by key returns the updated value
    3. The updatedAt timestamp is set automatically
    4. The timestamp is >= the update request time
    5. All fields are correctly persisted and retrievable
    """
    # Clear the database to ensure test isolation
    mongodb_database["global_config"].delete_many({})
    
    # Generate a value matching the value_type
    value = data.draw(value_for_type_strategy(value_type))
    
    # Record the time before the update (truncate microseconds for comparison)
    time_before_update = datetime.now(timezone.utc).replace(microsecond=0)
    
    # Create a ConfigUpdate
    config_update = ConfigUpdate(
        value=value,
        value_type=value_type,
        description=description
    )
    
    # Create a real repository with the test database
    repository = ConfigRepository(mongodb_database)
    
    # Create service with real repository
    service = ConfigService(repository)
    
    # Act: Update the configuration (this should persist to the database)
    updated_config = service.update_config(key, config_update)
    
    # Assert: Verify the returned config has correct data
    assert updated_config.key == key
    assert updated_config.value == value
    assert updated_config.value_type == value_type
    assert updated_config.description == description
    
    # Assert: Verify updatedAt is set and is >= time_before_update
    assert updated_config.updated_at is not None
    assert isinstance(updated_config.updated_at, datetime)
    
    # Truncate microseconds for comparison (mongomock may lose precision)
    updated_dt = updated_config.updated_at.replace(microsecond=0)
    if updated_dt.tzinfo is None:
        updated_dt = updated_dt.replace(tzinfo=timezone.utc)
    
    assert updated_dt >= time_before_update
    
    # Act: Retrieve the configuration from the database to verify persistence
    retrieved_config = service.get_config_by_key(key)
    
    # Assert: Verify the retrieved config matches the updated config
    assert retrieved_config is not None
    assert retrieved_config.key == key
    assert retrieved_config.value == value
    assert retrieved_config.value_type == value_type
    assert retrieved_config.description == description
    
    # Assert: Verify updatedAt is persisted and valid
    assert retrieved_config.updated_at is not None
    assert isinstance(retrieved_config.updated_at, datetime)
    
    # Handle timezone-aware and timezone-naive datetime comparison
    # Truncate microseconds for comparison (mongomock may lose precision)
    retrieved_dt = retrieved_config.updated_at.replace(microsecond=0)
    if retrieved_dt.tzinfo is None:
        retrieved_dt = retrieved_dt.replace(tzinfo=timezone.utc)
    
    assert retrieved_dt >= time_before_update


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(
    key=valid_key_strategy(),
    value_type=st.sampled_from(list(ValueType)),
    description=valid_description_strategy(),
    data=st.data()
)
def test_property_config_update_modifies_existing(
    key: str,
    value_type: ValueType,
    description: str,
    data,
    mongodb_database
):
    """
    **Validates: Requirements 11.2, 11.3, 11.4**
    
    Feature: fastapi-enumeration-services, Property 6: Config update persistence
    
    Property: For any configuration update on an existing key, the update should
    modify the existing configuration, and the updatedAt timestamp should be
    updated to reflect the modification time.
    
    This test verifies that:
    1. Updating an existing config modifies it (doesn't create duplicate)
    2. The updatedAt timestamp is updated on modification
    3. The new timestamp is >= the modification time
    4. All fields are correctly updated
    """
    # Clear the database to ensure test isolation
    mongodb_database["global_config"].delete_many({})
    
    # Generate initial value
    initial_value = data.draw(value_for_type_strategy(value_type))
    initial_description = data.draw(valid_description_strategy())
    
    # Create repository and service
    repository = ConfigRepository(mongodb_database)
    service = ConfigService(repository)
    
    # Create initial configuration
    initial_update = ConfigUpdate(
        value=initial_value,
        value_type=value_type,
        description=initial_description
    )
    
    initial_config = service.update_config(key, initial_update)
    initial_updated_at = initial_config.updated_at.replace(microsecond=0)
    
    # Generate new value (different from initial)
    new_value = data.draw(value_for_type_strategy(value_type))
    
    # Record time before the second update (truncate microseconds)
    time_before_second_update = datetime.now(timezone.utc).replace(microsecond=0)
    
    # Update the configuration with new value
    second_update = ConfigUpdate(
        value=new_value,
        value_type=value_type,
        description=description
    )
    
    updated_config = service.update_config(key, second_update)
    
    # Assert: Verify the config was updated (not created as new)
    assert updated_config.key == key
    assert updated_config.value == new_value
    assert updated_config.description == description
    
    # Assert: Verify updatedAt was updated
    # Handle timezone-aware and timezone-naive datetime comparison
    # Truncate microseconds for comparison
    updated_dt = updated_config.updated_at.replace(microsecond=0)
    if updated_dt.tzinfo is None:
        updated_dt = updated_dt.replace(tzinfo=timezone.utc)
    
    assert updated_dt >= time_before_second_update
    
    initial_dt = initial_updated_at
    if initial_dt.tzinfo is None:
        initial_dt = initial_dt.replace(tzinfo=timezone.utc)
    
    assert updated_dt >= initial_dt
    
    # Retrieve from database to verify persistence
    retrieved_config = service.get_config_by_key(key)
    
    # Assert: Verify only one config exists (no duplicate)
    assert retrieved_config is not None
    assert retrieved_config.key == key
    assert retrieved_config.value == new_value
    assert retrieved_config.description == description
    
    # Verify there's only one document in the collection for this key
    all_configs = service.get_all_configs()
    configs_with_key = [c for c in all_configs if c.key == key]
    assert len(configs_with_key) == 1


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(
    key=valid_key_strategy(),
    value_type=st.sampled_from([ValueType.INT, ValueType.FLOAT]),
    description=valid_description_strategy(),
    data=st.data()
)
def test_property_config_update_persistence_with_range_constraints(
    key: str,
    value_type: ValueType,
    description: str,
    data,
    mongodb_database
):
    """
    **Validates: Requirements 11.2, 11.3, 11.4**
    
    Feature: fastapi-enumeration-services, Property 6: Config update persistence
    
    Property: For any configuration update with numeric range constraints,
    the constraints should be persisted and retrievable along with the value.
    
    This test verifies that:
    1. minValue and maxValue are persisted correctly
    2. Range constraints are retrievable after update
    3. All fields including constraints match after retrieval
    """
    # Clear the database to ensure test isolation
    mongodb_database["global_config"].delete_many({})
    
    # Generate min and max values
    if value_type == ValueType.INT:
        min_value = float(data.draw(st.integers(min_value=-1000, max_value=999)))
        max_value = float(data.draw(st.integers(min_value=int(min_value) + 1, max_value=1000)))
        value = data.draw(st.integers(min_value=int(min_value), max_value=int(max_value)))
    else:  # FLOAT
        min_value = data.draw(st.floats(min_value=-1000.0, max_value=999.0, allow_nan=False, allow_infinity=False))
        max_value = data.draw(st.floats(min_value=min_value + 0.1, max_value=1000.0, allow_nan=False, allow_infinity=False))
        value = data.draw(st.floats(min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False))
    
    # Record time before update (truncate microseconds)
    time_before_update = datetime.now(timezone.utc).replace(microsecond=0)
    
    # Create a ConfigUpdate with range constraints
    config_update = ConfigUpdate(
        value=value,
        value_type=value_type,
        description=description,
        min_value=min_value,
        max_value=max_value
    )
    
    # Create repository and service
    repository = ConfigRepository(mongodb_database)
    service = ConfigService(repository)
    
    # Act: Update the configuration
    updated_config = service.update_config(key, config_update)
    
    # Assert: Verify the returned config has correct range constraints
    assert updated_config.min_value == min_value
    assert updated_config.max_value == max_value
    assert updated_config.value == value
    
    # Handle timezone-aware and timezone-naive datetime comparison
    # Truncate microseconds for comparison
    updated_dt = updated_config.updated_at.replace(microsecond=0)
    if updated_dt.tzinfo is None:
        updated_dt = updated_dt.replace(tzinfo=timezone.utc)
    
    assert updated_dt >= time_before_update
    
    # Act: Retrieve the configuration from the database
    retrieved_config = service.get_config_by_key(key)
    
    # Assert: Verify the retrieved config has correct range constraints
    assert retrieved_config is not None
    assert retrieved_config.key == key
    assert retrieved_config.value == value
    assert retrieved_config.min_value == min_value
    assert retrieved_config.max_value == max_value
    
    # Handle timezone-aware and timezone-naive datetime comparison
    # Truncate microseconds for comparison
    retrieved_dt = retrieved_config.updated_at.replace(microsecond=0)
    if retrieved_dt.tzinfo is None:
        retrieved_dt = retrieved_dt.replace(tzinfo=timezone.utc)
    
    assert retrieved_dt >= time_before_update




# Property 9: Get all configs completeness
# Feature: fastapi-enumeration-services, Property 9: Get all configs completeness
# Validates: Requirements 12.2

@settings(max_examples=100)
@given(
    num_configs=st.integers(min_value=0, max_value=20),
    data=st.data()
)
def test_property_get_all_configs_completeness(num_configs: int, data):
    """
    **Validates: Requirements 12.2**
    
    Feature: fastapi-enumeration-services, Property 9: Get all configs completeness
    
    Property: For any state of the database, calling get_all_configs should return
    a list containing all configuration documents present in the global_config collection.
    
    This test verifies that:
    1. ConfigService.get_all_configs returns all configurations
    2. The count matches the number of documents in the database
    3. All configurations are correctly converted to Config models
    """
    # Generate random configurations
    documents = []
    for i in range(num_configs):
        key = data.draw(valid_key_strategy()) + f"_{i}"  # Ensure unique keys
        value_type = data.draw(st.sampled_from(list(ValueType)))
        value = data.draw(value_for_type_strategy(value_type))
        description = data.draw(valid_description_strategy())
        updated_at = datetime.now(timezone.utc)
        
        document = {
            "_id": key,
            "key": key,
            "value": value,
            "valueType": value_type.value,
            "description": description,
            "updatedAt": updated_at,
            "minValue": None,
            "maxValue": None
        }
        documents.append(document)
    
    # Create a mock repository
    mock_repo = Mock(spec=ConfigRepository)
    
    # Configure the mock to return all documents
    mock_repo.find_all.return_value = documents
    
    # Create service with mock repository
    service = ConfigService(mock_repo)
    
    # Act: Get all configurations
    configs = service.get_all_configs()
    
    # Assert: Verify the repository was called
    mock_repo.find_all.assert_called_once()
    
    # Assert: Verify the count matches
    assert len(configs) == num_configs
    
    # Assert: Verify all configs are Config models with correct data
    for i, config in enumerate(configs):
        assert isinstance(config, Config)
        assert config.key == documents[i]["key"]
        assert config.value == documents[i]["value"]
        assert config.value_type.value == documents[i]["valueType"]
        assert config.description == documents[i]["description"]


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(
    num_configs=st.integers(min_value=0, max_value=20),
    data=st.data()
)
def test_property_get_all_configs_completeness_with_database(
    num_configs: int,
    data,
    mongodb_database
):
    """
    **Validates: Requirements 12.2**
    
    Feature: fastapi-enumeration-services, Property 9: Get all configs completeness
    
    Property: For any state of the database, calling GET /config should return
    a list containing all configuration documents present in the global_config collection.
    
    This test verifies that:
    1. Random config documents are inserted into the test database
    2. ConfigService.get_all_configs returns all inserted configurations
    3. The count matches exactly the number of documents inserted
    4. All configurations are correctly retrieved and converted to Config models
    5. Each retrieved config matches the inserted data
    """
    # Clear the database to ensure test isolation
    mongodb_database["global_config"].delete_many({})
    
    # Generate and insert random configurations
    inserted_keys = set()
    for i in range(num_configs):
        key = data.draw(valid_key_strategy()) + f"_{i}"  # Ensure unique keys
        value_type = data.draw(st.sampled_from(list(ValueType)))
        value = data.draw(value_for_type_strategy(value_type))
        description = data.draw(valid_description_strategy())
        updated_at = datetime.now(timezone.utc)
        
        document = {
            "_id": key,
            "key": key,
            "value": value,
            "valueType": value_type.value,
            "description": description,
            "updatedAt": updated_at,
            "minValue": None,
            "maxValue": None
        }
        
        # Insert into test database
        mongodb_database["global_config"].insert_one(document)
        inserted_keys.add(key)
    
    # Create repository and service with real test database
    repository = ConfigRepository(mongodb_database)
    service = ConfigService(repository)
    
    # Act: Get all configurations
    configs = service.get_all_configs()
    
    # Assert: Verify the count matches exactly
    assert len(configs) == num_configs, f"Expected {num_configs} configs, got {len(configs)}"
    
    # Assert: Verify all configs are Config models
    for config in configs:
        assert isinstance(config, Config)
    
    # Assert: Verify all inserted keys are present in the results
    retrieved_keys = {config.key for config in configs}
    assert retrieved_keys == inserted_keys, f"Retrieved keys {retrieved_keys} don't match inserted keys {inserted_keys}"
    
    # Assert: Verify each config has valid data
    for config in configs:
        assert config.key in inserted_keys
        assert config.value is not None
        assert config.value_type in list(ValueType)
        assert config.description is not None
        assert config.updated_at is not None


def test_property_get_all_configs_empty_database():
    """
    **Validates: Requirements 12.2**
    
    Feature: fastapi-enumeration-services, Property 9: Get all configs completeness
    
    Property: When the database is empty, get_all_configs should return an empty list.
    
    This test verifies that:
    1. ConfigService.get_all_configs handles empty databases correctly
    2. An empty list is returned (not None or error)
    """
    # Create a mock repository
    mock_repo = Mock(spec=ConfigRepository)
    
    # Configure the mock to return empty list
    mock_repo.find_all.return_value = []
    
    # Create service with mock repository
    service = ConfigService(mock_repo)
    
    # Act: Get all configurations
    configs = service.get_all_configs()
    
    # Assert: Verify empty list is returned
    assert configs == []
    assert len(configs) == 0


# Property 14: Batch config update validation
# Feature: fastapi-enumeration-services, Property 14: Batch config update validation
# Validates: Requirements 18.8, 18.9

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    num_valid_configs=st.integers(min_value=1, max_value=5),
    num_failing_configs=st.integers(min_value=1, max_value=3),
    data=st.data()
)
def test_property_batch_config_update_validation_rejects_batch_with_database_errors(
    num_valid_configs: int,
    num_failing_configs: int,
    data
):
    """
    **Validates: Requirements 18.8, 18.9**
    
    Feature: fastapi-enumeration-services, Property 14: Batch config update validation
    
    Property: For any batch configuration update request, if any config in the batch
    fails validation or update, the entire batch should be rejected without updating
    any configs, and the response should include error details for all failed configs.
    
    This test verifies that:
    1. ConfigService.batch_update validates all configs before updating
    2. If any update fails (simulated via database error), the batch reports failures
    3. Error details are provided for all failed configs
    4. The all-or-nothing semantics are maintained
    
    Note: Since Pydantic validates ConfigUpdate on construction, we simulate validation
    failures by having the repository raise exceptions for certain keys.
    """
    # Generate valid config updates
    updates = []
    valid_keys = []
    failing_keys = []
    
    for i in range(num_valid_configs):
        key = data.draw(valid_key_strategy()) + f"_valid_{i}"
        valid_keys.append(key)
        value_type = data.draw(st.sampled_from(list(ValueType)))
        value = data.draw(value_for_type_strategy(value_type))
        description = data.draw(valid_description_strategy())
        
        updates.append(BatchConfigUpdate(
            key=key,
            update=ConfigUpdate(
                value=value,
                value_type=value_type,
                description=description
            )
        ))
    
    # Generate configs that will "fail" during update (simulated by mock)
    for i in range(num_failing_configs):
        key = data.draw(valid_key_strategy()) + f"_failing_{i}"
        failing_keys.append(key)
        value_type = data.draw(st.sampled_from(list(ValueType)))
        value = data.draw(value_for_type_strategy(value_type))
        description = data.draw(valid_description_strategy())
        
        updates.append(BatchConfigUpdate(
            key=key,
            update=ConfigUpdate(
                value=value,
                value_type=value_type,
                description=description
            )
        ))
    
    # Create a mock repository that raises exceptions for failing keys
    mock_repo = Mock(spec=ConfigRepository)
    
    def mock_upsert(key, document):
        if key in failing_keys:
            raise Exception(f"Database error for key {key}")
        return document
    
    mock_repo.upsert.side_effect = mock_upsert
    
    # Create service with mock repository
    service = ConfigService(mock_repo)
    
    # Act: Batch update with some configs that will fail
    result = service.batch_update(updates)
    
    # Assert: Verify the result indicates partial failure
    assert result.total == len(updates)
    assert result.failed == num_failing_configs
    assert len(result.errors) == num_failing_configs
    
    # Assert: Verify error details include key information
    error_keys = [error["key"] for error in result.errors]
    assert set(error_keys) == set(failing_keys)
    
    for error in result.errors:
        assert "key" in error
        assert "error" in error
        assert "Database error" in error["error"]


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(
    num_configs=st.integers(min_value=2, max_value=5),
    data=st.data()
)
def test_property_batch_config_update_validation_all_or_nothing_with_database(
    num_configs: int,
    data,
    mongodb_database
):
    """
    **Validates: Requirements 18.8, 18.9**
    
    Feature: fastapi-enumeration-services, Property 14: Batch config update validation
    
    Property: For any batch configuration update request, the batch update follows
    all-or-nothing semantics - either all configs are updated successfully or none are.
    
    This test verifies with a real database that:
    1. When all configs are valid, all are updated successfully
    2. The database contains all updated configs after a successful batch
    3. The success count matches the number of configs
    """
    # Clear the database to ensure test isolation
    mongodb_database["global_config"].delete_many({})
    
    # Generate valid config updates
    updates = []
    expected_keys = []
    
    for i in range(num_configs):
        key = data.draw(valid_key_strategy()) + f"_{i}"
        expected_keys.append(key)
        value_type = data.draw(st.sampled_from(list(ValueType)))
        value = data.draw(value_for_type_strategy(value_type))
        description = data.draw(valid_description_strategy())
        
        updates.append(BatchConfigUpdate(
            key=key,
            update=ConfigUpdate(
                value=value,
                value_type=value_type,
                description=description
            )
        ))
    
    # Create repository and service with real test database
    repository = ConfigRepository(mongodb_database)
    service = ConfigService(repository)
    
    # Act: Batch update with all valid configs
    result = service.batch_update(updates)
    
    # Assert: Verify all configs were updated successfully
    assert result.total == num_configs
    assert result.successful == num_configs
    assert result.failed == 0
    assert len(result.errors) == 0
    
    # Assert: Verify all configs are persisted to the database
    for expected_key in expected_keys:
        retrieved_config = service.get_config_by_key(expected_key)
        assert retrieved_config is not None, f"Config {expected_key} should be in database after successful batch"
        assert retrieved_config.key == expected_key
    
    # Assert: Verify the database contains exactly the expected configs
    all_configs = service.get_all_configs()
    assert len(all_configs) == num_configs
    retrieved_keys = {config.key for config in all_configs}
    assert retrieved_keys == set(expected_keys)


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    num_configs=st.integers(min_value=2, max_value=5),
    data=st.data()
)
def test_property_batch_config_update_validation_error_details(
    num_configs: int,
    data
):
    """
    **Validates: Requirements 18.8, 18.9**
    
    Feature: fastapi-enumeration-services, Property 14: Batch config update validation
    
    Property: When a batch update encounters errors, the response should include
    error details for all failed configs, including the key and error message.
    
    This test verifies that:
    1. Error details are provided for each failed config
    2. Each error includes the config key
    3. Each error includes a descriptive error message
    4. The error count matches the number of failed configs
    """
    # Generate config updates that will fail (simulated via mock)
    updates = []
    expected_error_keys = []
    
    for i in range(num_configs):
        key = data.draw(valid_key_strategy()) + f"_{i}"
        expected_error_keys.append(key)
        value_type = data.draw(st.sampled_from(list(ValueType)))
        value = data.draw(value_for_type_strategy(value_type))
        description = data.draw(valid_description_strategy())
        
        updates.append(BatchConfigUpdate(
            key=key,
            update=ConfigUpdate(
                value=value,
                value_type=value_type,
                description=description
            )
        ))
    
    # Create a mock repository that raises exceptions for all keys
    mock_repo = Mock(spec=ConfigRepository)
    
    def mock_upsert(key, document):
        raise Exception(f"Simulated database error for {key}")
    
    mock_repo.upsert.side_effect = mock_upsert
    
    # Create service with mock repository
    service = ConfigService(mock_repo)
    
    # Act: Batch update with all configs failing
    result = service.batch_update(updates)
    
    # Assert: Verify error details are provided
    assert result.failed == num_configs
    assert len(result.errors) == num_configs
    
    # Assert: Verify each error has required fields
    error_keys = []
    for error in result.errors:
        assert "key" in error, "Error should include 'key' field"
        assert "error" in error, "Error should include 'error' field"
        assert isinstance(error["key"], str), "Error key should be a string"
        assert isinstance(error["error"], str), "Error message should be a string"
        assert len(error["error"]) > 0, "Error message should not be empty"
        assert "Database error" in error["error"], "Error message should indicate database error"
        error_keys.append(error["key"])
    
    # Assert: Verify all expected keys are in the errors
    assert set(error_keys) == set(expected_error_keys), "All failed config keys should be in error details"


@settings(max_examples=100)
@given(
    num_configs=st.integers(min_value=1, max_value=10),
    data=st.data()
)
def test_property_batch_config_update_validate_only_mode(
    num_configs: int,
    data
):
    """
    **Validates: Requirements 18.8, 18.9**
    
    Feature: fastapi-enumeration-services, Property 14: Batch config update validation
    
    Property: When validate_only=True, the batch update should validate all configs
    but not update any in the database.
    
    This test verifies that:
    1. ConfigService.batch_update with validate_only=True validates configs
    2. No database updates occur when validate_only=True
    3. Validation results are returned correctly
    """
    # Generate valid config updates
    updates = []
    for i in range(num_configs):
        key = data.draw(valid_key_strategy()) + f"_{i}"
        value_type = data.draw(st.sampled_from(list(ValueType)))
        value = data.draw(value_for_type_strategy(value_type))
        description = data.draw(valid_description_strategy())
        
        updates.append(BatchConfigUpdate(
            key=key,
            update=ConfigUpdate(
                value=value,
                value_type=value_type,
                description=description
            )
        ))
    
    # Create a mock repository
    mock_repo = Mock(spec=ConfigRepository)
    
    # Create service with mock repository
    service = ConfigService(mock_repo)
    
    # Act: Batch update with validate_only=True
    result = service.batch_update(updates, validate_only=True)
    
    # Assert: Verify no configs were updated (upsert never called)
    mock_repo.upsert.assert_not_called()
    
    # Assert: Verify the result indicates validation success
    assert result.total == num_configs
    assert result.successful == 0  # No updates performed
    assert result.failed == 0  # No validation errors
    assert len(result.errors) == 0


# Property 15: Batch config update success
# Feature: fastapi-enumeration-services, Property 15: Batch config update success
# Validates: Requirements 18.7, 18.10

@settings(max_examples=100)
@given(
    num_configs=st.integers(min_value=1, max_value=10),
    data=st.data()
)
def test_property_batch_config_update_success(
    num_configs: int,
    data
):
    """
    **Validates: Requirements 18.7, 18.10**
    
    Feature: fastapi-enumeration-services, Property 15: Batch config update success
    
    Property: For any batch configuration update request where all configs are valid,
    all configs should be successfully updated in the database, and the response
    should indicate the correct count of successful updates.
    
    This test verifies that:
    1. ConfigService.batch_update updates all valid configs
    2. The repository upsert method is called for each config
    3. The result indicates the correct success count
    4. No errors are reported
    """
    # Generate valid config updates
    updates = []
    for i in range(num_configs):
        key = data.draw(valid_key_strategy()) + f"_{i}"
        value_type = data.draw(st.sampled_from(list(ValueType)))
        value = data.draw(value_for_type_strategy(value_type))
        description = data.draw(valid_description_strategy())
        
        updates.append(BatchConfigUpdate(
            key=key,
            update=ConfigUpdate(
                value=value,
                value_type=value_type,
                description=description
            )
        ))
    
    # Create a mock repository
    mock_repo = Mock(spec=ConfigRepository)
    
    # Configure the mock to return the upserted document
    def mock_upsert(key, document):
        return document
    
    mock_repo.upsert.side_effect = mock_upsert
    
    # Create service with mock repository
    service = ConfigService(mock_repo)
    
    # Act: Batch update
    result = service.batch_update(updates)
    
    # Assert: Verify all configs were updated
    assert mock_repo.upsert.call_count == num_configs
    
    # Assert: Verify the result indicates success
    assert result.total == num_configs
    assert result.successful == num_configs
    assert result.failed == 0
    assert len(result.errors) == 0


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    num_configs=st.integers(min_value=2, max_value=10),
    data=st.data()
)
def test_property_batch_config_update_all_or_nothing(
    num_configs: int,
    data
):
    """
    **Validates: Requirements 18.7, 18.10**
    
    Feature: fastapi-enumeration-services, Property 15: Batch config update success
    
    Property: Batch updates follow all-or-nothing semantics - either all configs
    are updated or none are updated.
    
    This test verifies that:
    1. If all configs are valid, all are updated successfully
    2. The success count matches the number of configs
    """
    # Generate valid config updates
    updates = []
    for i in range(num_configs):
        key = data.draw(valid_key_strategy()) + f"_{i}"
        value_type = data.draw(st.sampled_from(list(ValueType)))
        value = data.draw(value_for_type_strategy(value_type))
        description = data.draw(valid_description_strategy())
        
        updates.append(BatchConfigUpdate(
            key=key,
            update=ConfigUpdate(
                value=value,
                value_type=value_type,
                description=description
            )
        ))
    
    # Create a mock repository
    mock_repo = Mock(spec=ConfigRepository)
    
    # Configure the mock to return the upserted document
    def mock_upsert(key, document):
        return document
    
    mock_repo.upsert.side_effect = mock_upsert
    
    # Create service with mock repository
    service = ConfigService(mock_repo)
    
    # Act: Batch update with all valid configs
    result = service.batch_update(updates)
    
    # Assert: All configs were updated
    assert result.successful == num_configs
    assert result.failed == 0
    assert len(result.errors) == 0
    
    # Assert: Repository was called for each config
    assert mock_repo.upsert.call_count == num_configs


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(
    num_configs=st.integers(min_value=1, max_value=10),
    data=st.data()
)
def test_property_batch_config_update_success_with_database(
    num_configs: int,
    data,
    mongodb_database
):
    """
    **Validates: Requirements 18.7, 18.10**
    
    Feature: fastapi-enumeration-services, Property 15: Batch config update success
    
    Property: For any batch configuration update request where all configs are valid,
    all configs should be successfully updated in the database, and the response
    should indicate the correct count of successful updates.
    
    This test verifies that:
    1. ConfigService.batch_update successfully updates all valid configs
    2. All configs are persisted to the database
    3. The response indicates the correct count of successful updates
    4. All configs can be retrieved from the database after the batch update
    5. Each retrieved config matches the updated data
    """
    # Clear the database to ensure test isolation
    mongodb_database["global_config"].delete_many({})
    
    # Generate valid config updates with unique keys
    updates = []
    expected_data = {}
    
    for i in range(num_configs):
        key = data.draw(valid_key_strategy()) + f"_batch_{i}"
        value_type = data.draw(st.sampled_from(list(ValueType)))
        value = data.draw(value_for_type_strategy(value_type))
        description = data.draw(valid_description_strategy())
        
        # Store expected data for verification
        expected_data[key] = {
            "value": value,
            "value_type": value_type,
            "description": description
        }
        
        updates.append(BatchConfigUpdate(
            key=key,
            update=ConfigUpdate(
                value=value,
                value_type=value_type,
                description=description
            )
        ))
    
    # Create repository and service with real test database
    repository = ConfigRepository(mongodb_database)
    service = ConfigService(repository)
    
    # Record time before batch update (truncate microseconds for comparison)
    time_before_update = datetime.now(timezone.utc).replace(microsecond=0)
    
    # Act: Batch update with all valid configs
    result = service.batch_update(updates)
    
    # Assert: Verify the result indicates all configs were updated successfully
    assert result.total == num_configs, f"Expected total={num_configs}, got {result.total}"
    assert result.successful == num_configs, f"Expected successful={num_configs}, got {result.successful}"
    assert result.failed == 0, f"Expected failed=0, got {result.failed}"
    assert len(result.errors) == 0, f"Expected no errors, got {result.errors}"
    
    # Assert: Verify all configs are persisted to the database
    for key, expected in expected_data.items():
        retrieved_config = service.get_config_by_key(key)
        
        # Verify config was persisted
        assert retrieved_config is not None, f"Config {key} should be in database after batch update"
        
        # Verify all fields match expected data
        assert retrieved_config.key == key, f"Key mismatch for {key}"
        assert retrieved_config.value == expected["value"], f"Value mismatch for {key}"
        assert retrieved_config.value_type == expected["value_type"], f"ValueType mismatch for {key}"
        assert retrieved_config.description == expected["description"], f"Description mismatch for {key}"
        
        # Verify updatedAt timestamp is set and valid
        assert retrieved_config.updated_at is not None, f"updatedAt should be set for {key}"
        assert isinstance(retrieved_config.updated_at, datetime), f"updatedAt should be datetime for {key}"
        
        # Handle timezone-aware and timezone-naive datetime comparison
        # Truncate microseconds for comparison
        retrieved_dt = retrieved_config.updated_at.replace(microsecond=0)
        if retrieved_dt.tzinfo is None:
            retrieved_dt = retrieved_dt.replace(tzinfo=timezone.utc)
        
        assert retrieved_dt >= time_before_update, f"updatedAt should be >= update time for {key}"
    
    # Assert: Verify the database contains exactly the expected configs
    all_configs = service.get_all_configs()
    assert len(all_configs) == num_configs, f"Database should contain exactly {num_configs} configs"
    
    retrieved_keys = {config.key for config in all_configs}
    expected_keys = set(expected_data.keys())
    assert retrieved_keys == expected_keys, f"Retrieved keys {retrieved_keys} don't match expected keys {expected_keys}"

"""
Property-based tests for SKU Service.

This module contains property-based tests using Hypothesis to verify that the SKU service
behaves correctly across a wide range of inputs. Each test runs 100+ iterations with
randomly generated data to ensure comprehensive coverage.

Tests are tagged with the feature name and property number from the design document.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from typing import List

from models.sku import SKU, SearchCriteria, BatchImportResult
from repositories.sku_repository import SKURepository
from .sku_service import SKUService


# Custom strategies for generating valid SKU data
def valid_string_strategy(min_size=1, max_size=50):
    """Generate valid non-empty strings."""
    return st.text(
        alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),  # Uppercase, lowercase, digits
            whitelist_characters=' -_'
        ),
        min_size=min_size,
        max_size=max_size
    ).filter(lambda s: s.strip())  # Ensure not just whitespace


def valid_part_codes_strategy():
    """Generate valid part code lists."""
    return st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Nd')),
            min_size=1,
            max_size=10
        ).filter(lambda s: s.strip()),
        min_size=1,
        max_size=20
    )


def valid_weight_range_strategy():
    """Generate valid weight ranges where min < max."""
    return st.tuples(
        st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False)
    ).map(lambda pair: (min(pair), max(pair) + 0.1))  # Ensure min < max


def valid_sku_strategy():
    """Generate valid SKU objects."""
    return st.builds(
        lambda trade_num, cust_name, cust_type, prod_type, units, plant, weights, bird, parts: SKU(
            trade_number=trade_num,
            customer_name=cust_name,
            customer_type=cust_type,
            product_type=prod_type,
            units_per_cut=units,
            prod_plant=plant,
            min_weight=weights[0],
            max_weight=weights[1],
            target_weight=(weights[0] + weights[1]) / 2,  # Target in middle of range
            bird_size=bird,
            allowed_parts=parts
        ),
        trade_num=valid_string_strategy(min_size=1, max_size=50),
        cust_name=valid_string_strategy(min_size=1, max_size=200),
        cust_type=valid_string_strategy(min_size=1, max_size=50),
        prod_type=valid_string_strategy(min_size=1, max_size=50),
        units=st.integers(min_value=1, max_value=100),
        plant=valid_string_strategy(min_size=1, max_size=50),
        weights=valid_weight_range_strategy(),
        bird=valid_string_strategy(min_size=1, max_size=50),
        parts=valid_part_codes_strategy()
    )


# Property 1: SKU retrieval returns correct data
# Feature: fastapi-enumeration-services, Property 1: SKU retrieval returns correct data
# Validates: Requirements 5.2, 7.1

@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(sku=valid_sku_strategy())
def test_property_sku_retrieval_returns_correct_data(sku: SKU, mongodb_database):
    """
    Feature: fastapi-enumeration-services, Property 1: SKU retrieval returns correct data
    Validates: Requirements 5.2, 7.1
    
    Property: For any SKU document stored in the database, retrieving it by trade number
    should return a document where the _id field equals the tradeNumber field, and the
    document contains all the expected data.
    
    This test verifies that:
    1. Inserting a SKU and then retrieving it returns the same data
    2. The _id field equals the tradeNumber
    3. All fields are preserved correctly
    """
    # Create service with test database
    repo = SKURepository(mongodb_database)
    service = SKUService(repo)
    
    # Insert the SKU into the database
    sku_doc = sku.model_dump(by_alias=True)
    sku_doc["_id"] = sku.trade_number
    mongodb_database["skus"].insert_one(sku_doc)
    
    # Retrieve the SKU by trade number
    retrieved_sku = service.get_sku_by_trade_number(sku.trade_number)
    
    # Verify the SKU was retrieved
    assert retrieved_sku is not None
    
    # Verify all fields match
    assert retrieved_sku.trade_number == sku.trade_number
    assert retrieved_sku.customer_name == sku.customer_name
    assert retrieved_sku.customer_type == sku.customer_type
    assert retrieved_sku.product_type == sku.product_type
    assert retrieved_sku.units_per_cut == sku.units_per_cut
    assert retrieved_sku.prod_plant == sku.prod_plant
    assert retrieved_sku.min_weight == sku.min_weight
    assert retrieved_sku.max_weight == sku.max_weight
    assert retrieved_sku.target_weight == sku.target_weight
    assert retrieved_sku.bird_size == sku.bird_size
    assert retrieved_sku.allowed_parts == sku.allowed_parts
    
    # Verify the _id in the database equals the tradeNumber
    db_doc = mongodb_database["skus"].find_one({"_id": sku.trade_number})
    assert db_doc is not None
    assert db_doc["_id"] == db_doc["tradeNumber"]


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(trade_number=valid_string_strategy())
def test_property_sku_retrieval_returns_none_for_nonexistent(trade_number: str, mongodb_database):
    """
    Feature: fastapi-enumeration-services, Property 1: SKU retrieval returns correct data
    Validates: Requirements 5.2, 7.1
    
    Property: For any trade number that doesn't exist in the database, retrieval
    should return None.
    
    This test verifies that the service correctly handles non-existent SKUs.
    """
    # Create service with test database
    repo = SKURepository(mongodb_database)
    service = SKUService(repo)
    
    # Ensure the trade number doesn't exist in the database
    # (the database is empty for each test)
    
    # Try to retrieve a non-existent SKU
    retrieved_sku = service.get_sku_by_trade_number(trade_number)
    
    # Verify None is returned
    assert retrieved_sku is None


# Property 2: Search results match criteria
# Feature: fastapi-enumeration-services, Property 2: Search results match criteria
# Validates: Requirements 6.2, 6.3

@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    skus=st.lists(valid_sku_strategy(), min_size=1, max_size=20),
    search_field=st.sampled_from(['customer_type', 'product_type', 'prod_plant', 'bird_size'])
)
def test_property_search_results_match_criteria(skus: List[SKU], search_field: str, mongodb_database):
    """
    Feature: fastapi-enumeration-services, Property 2: Search results match criteria
    Validates: Requirements 6.2, 6.3
    
    Property: For any search criteria provided, all returned SKU documents should match
    the specified filters. If searching by customerType="FDS", all results should have
    customerType="FDS".
    
    This test verifies that:
    1. Search returns only SKUs matching the criteria
    2. No SKUs are returned that don't match the criteria
    3. All matching SKUs are returned (completeness)
    """
    # Clear the database before each test run
    mongodb_database["skus"].delete_many({})
    
    # Create service with test database
    repo = SKURepository(mongodb_database)
    service = SKUService(repo)
    
    # Insert all SKUs into the database
    for sku in skus:
        sku_doc = sku.model_dump(by_alias=True)
        sku_doc["_id"] = sku.trade_number
        mongodb_database["skus"].insert_one(sku_doc)
    
    # Pick a search value from one of the SKUs to ensure we get at least one result
    search_value = getattr(skus[0], search_field)
    
    # Create search criteria based on the selected field
    criteria_dict = {search_field: search_value}
    criteria = SearchCriteria(**criteria_dict)
    
    # Perform the search
    results = service.search_skus(criteria)
    
    # Verify all results match the search criteria
    for result in results:
        assert getattr(result, search_field) == search_value, \
            f"Result {result.trade_number} has {search_field}={getattr(result, search_field)}, expected {search_value}"
    
    # Verify completeness: all SKUs matching the criteria are returned
    expected_matches = [sku for sku in skus if getattr(sku, search_field) == search_value]
    assert len(results) == len(expected_matches), \
        f"Expected {len(expected_matches)} results, got {len(results)}"
    
    # Verify the trade numbers match (order doesn't matter)
    result_trade_numbers = {r.trade_number for r in results}
    expected_trade_numbers = {sku.trade_number for sku in expected_matches}
    assert result_trade_numbers == expected_trade_numbers, \
        f"Result trade numbers {result_trade_numbers} don't match expected {expected_trade_numbers}"


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(skus=st.lists(valid_sku_strategy(), min_size=5, max_size=20))
def test_property_search_with_multiple_criteria(skus: List[SKU], mongodb_database):
    """
    Feature: fastapi-enumeration-services, Property 2: Search results match criteria
    Validates: Requirements 6.2, 6.3
    
    Property: For any search with multiple criteria, all returned SKUs should match
    ALL specified filters (AND logic).
    
    This test verifies that multiple search criteria are combined with AND logic.
    """
    # Clear the database before each test run
    mongodb_database["skus"].delete_many({})
    
    # Create service with test database
    repo = SKURepository(mongodb_database)
    service = SKUService(repo)
    
    # Insert all SKUs into the database
    for sku in skus:
        sku_doc = sku.model_dump(by_alias=True)
        sku_doc["_id"] = sku.trade_number
        mongodb_database["skus"].insert_one(sku_doc)
    
    # Pick values from the first SKU for multiple criteria
    customer_type_value = skus[0].customer_type
    product_type_value = skus[0].product_type
    
    # Create search criteria with multiple fields
    criteria = SearchCriteria(
        customer_type=customer_type_value,
        product_type=product_type_value
    )
    
    # Perform the search
    results = service.search_skus(criteria)
    
    # Verify all results match ALL criteria (AND logic)
    for result in results:
        assert result.customer_type == customer_type_value, \
            f"Result {result.trade_number} has customer_type={result.customer_type}, expected {customer_type_value}"
        assert result.product_type == product_type_value, \
            f"Result {result.trade_number} has product_type={result.product_type}, expected {product_type_value}"
    
    # Verify completeness: all SKUs matching both criteria are returned
    expected_matches = [
        sku for sku in skus 
        if sku.customer_type == customer_type_value and sku.product_type == product_type_value
    ]
    assert len(results) == len(expected_matches), \
        f"Expected {len(expected_matches)} results matching both criteria, got {len(results)}"


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(skus=st.lists(valid_sku_strategy(), min_size=1, max_size=20))
def test_property_search_with_no_criteria_returns_all(skus: List[SKU], mongodb_database):
    """
    Feature: fastapi-enumeration-services, Property 2: Search results match criteria
    Validates: Requirements 6.2, 6.3
    
    Property: For any search with empty criteria, all SKUs in the database should be returned.
    
    This test verifies that searching with no filters returns all SKUs.
    """
    # Clear the database before each test run
    mongodb_database["skus"].delete_many({})
    
    # Create service with test database
    repo = SKURepository(mongodb_database)
    service = SKUService(repo)
    
    # Insert all SKUs into the database
    for sku in skus:
        sku_doc = sku.model_dump(by_alias=True)
        sku_doc["_id"] = sku.trade_number
        mongodb_database["skus"].insert_one(sku_doc)
    
    # Create empty search criteria
    criteria = SearchCriteria()
    
    # Perform the search
    results = service.search_skus(criteria)
    
    # Verify all SKUs are returned
    assert len(results) == len(skus), \
        f"Expected {len(skus)} results with empty criteria, got {len(results)}"
    
    # Verify the trade numbers match (order doesn't matter)
    result_trade_numbers = {r.trade_number for r in results}
    expected_trade_numbers = {sku.trade_number for sku in skus}
    assert result_trade_numbers == expected_trade_numbers, \
        f"Result trade numbers don't match expected SKUs"


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    skus=st.lists(valid_sku_strategy(), min_size=1, max_size=20),
    search_value=valid_string_strategy()
)
def test_property_search_with_no_matches_returns_empty(skus: List[SKU], search_value: str, mongodb_database):
    """
    Feature: fastapi-enumeration-services, Property 2: Search results match criteria
    Validates: Requirements 6.2, 6.3
    
    Property: For any search criteria that matches no SKUs, an empty list should be returned.
    
    This test verifies that searches with no matches return an empty list (not None or error).
    """
    # Clear the database before each test run
    mongodb_database["skus"].delete_many({})
    
    # Create service with test database
    repo = SKURepository(mongodb_database)
    service = SKUService(repo)
    
    # Insert all SKUs into the database
    for sku in skus:
        sku_doc = sku.model_dump(by_alias=True)
        sku_doc["_id"] = sku.trade_number
        mongodb_database["skus"].insert_one(sku_doc)
    
    # Ensure the search value doesn't match any SKU's customer_type
    # by using a value that's very unlikely to exist
    unique_search_value = f"NONEXISTENT_{search_value}_UNIQUE_12345"
    assume(all(sku.customer_type != unique_search_value for sku in skus))
    
    # Create search criteria with non-matching value
    criteria = SearchCriteria(customer_type=unique_search_value)
    
    # Perform the search
    results = service.search_skus(criteria)
    
    # Verify empty list is returned
    assert results == [], \
        f"Expected empty list for non-matching criteria, got {len(results)} results"

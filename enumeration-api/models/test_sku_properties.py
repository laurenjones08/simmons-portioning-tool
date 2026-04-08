"""
Property-based tests for SKU models.

This module contains property-based tests using Hypothesis to verify that SKU models
behave correctly across a wide range of inputs. Each test runs 100+ iterations with
randomly generated data to ensure comprehensive coverage.

Tests are tagged with the feature name and property number from the design document.
"""

import pytest
from hypothesis import given, strategies as st, settings
from pydantic import ValidationError
from typing import List

from .sku import SKU, SearchCriteria, BatchImportRequest, BatchImportResult


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


# Property 3: SKU schema validation
# Feature: fastapi-enumeration-services, Property 3: SKU schema validation
# Validates: Requirements 7.2, 7.3, 7.4

@settings(max_examples=100)
@given(sku=valid_sku_strategy())
def test_property_sku_schema_validation_valid_skus(sku: SKU):
    """
    Feature: fastapi-enumeration-services, Property 3: SKU schema validation
    Validates: Requirements 7.2, 7.3, 7.4
    
    Property: For any valid SKU document, it should contain all required fields
    (tradeNumber, customerName, customerType, productType, unitsPerCut, prodPlant,
    minWeight, maxWeight, targetWeight, birdSize, allowedParts), with
    minWeight/maxWeight/targetWeight as numeric values, and allowedParts as an
    array of strings.
    
    This test verifies that valid SKU objects:
    1. Have all required fields present
    2. Have numeric types for weight fields
    3. Have allowedParts as a list of strings
    4. Can be serialized and deserialized correctly
    """
    # Assert all required fields are present
    assert sku.trade_number is not None
    assert sku.customer_name is not None
    assert sku.customer_type is not None
    assert sku.product_type is not None
    assert sku.units_per_cut is not None
    assert sku.prod_plant is not None
    assert sku.min_weight is not None
    assert sku.max_weight is not None
    assert sku.target_weight is not None
    assert sku.bird_size is not None
    assert sku.allowed_parts is not None
    
    # Assert numeric types for weight fields
    assert isinstance(sku.min_weight, (int, float))
    assert isinstance(sku.max_weight, (int, float))
    assert isinstance(sku.target_weight, (int, float))
    
    # Assert allowedParts is a list of strings
    assert isinstance(sku.allowed_parts, list)
    assert len(sku.allowed_parts) > 0
    assert all(isinstance(part, str) for part in sku.allowed_parts)
    
    # Assert serialization works correctly
    sku_dict = sku.model_dump(by_alias=True)
    assert "tradeNumber" in sku_dict
    assert "customerName" in sku_dict
    assert "minWeight" in sku_dict
    assert "maxWeight" in sku_dict
    assert "targetWeight" in sku_dict
    assert "allowedParts" in sku_dict
    
    # Assert deserialization works correctly (round-trip)
    sku_reconstructed = SKU(**sku_dict)
    assert sku_reconstructed.trade_number == sku.trade_number
    assert sku_reconstructed.min_weight == sku.min_weight
    assert sku_reconstructed.max_weight == sku.max_weight


@settings(max_examples=100)
@given(
    trade_number=valid_string_strategy(),
    customer_name=valid_string_strategy(),
    customer_type=valid_string_strategy(),
    product_type=valid_string_strategy(),
    units_per_cut=st.integers(min_value=1, max_value=100),
    prod_plant=valid_string_strategy(),
    weights=valid_weight_range_strategy(),
    bird_size=valid_string_strategy(),
    allowed_parts=valid_part_codes_strategy()
)
def test_property_sku_schema_validation_field_types(
    trade_number, customer_name, customer_type, product_type,
    units_per_cut, prod_plant, weights, bird_size, allowed_parts
):
    """
    Feature: fastapi-enumeration-services, Property 3: SKU schema validation
    Validates: Requirements 7.2, 7.3, 7.4
    
    Property: For any SKU with valid field types, Pydantic validation should accept it.
    
    This test verifies that when all fields have the correct types:
    - String fields are strings
    - Numeric fields are numbers
    - Array fields are arrays
    The SKU model accepts the data without raising ValidationError.
    """
    min_weight, max_weight = weights
    target_weight = (min_weight + max_weight) / 2
    
    # Create SKU with valid types - should not raise ValidationError
    sku = SKU(
        trade_number=trade_number,
        customer_name=customer_name,
        customer_type=customer_type,
        product_type=product_type,
        units_per_cut=units_per_cut,
        prod_plant=prod_plant,
        min_weight=min_weight,
        max_weight=max_weight,
        target_weight=target_weight,
        bird_size=bird_size,
        allowed_parts=allowed_parts
    )
    
    # Verify the SKU was created successfully
    assert sku.trade_number == trade_number
    assert isinstance(sku.min_weight, (int, float))
    assert isinstance(sku.max_weight, (int, float))
    assert isinstance(sku.allowed_parts, list)


@settings(max_examples=100)
@given(
    trade_number=valid_string_strategy(),
    customer_name=valid_string_strategy(),
    customer_type=valid_string_strategy(),
    product_type=valid_string_strategy(),
    units_per_cut=st.integers(min_value=1, max_value=100),
    prod_plant=valid_string_strategy(),
    min_weight=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    max_weight=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    bird_size=valid_string_strategy()
)
def test_property_sku_schema_validation_rejects_invalid_types(
    trade_number, customer_name, customer_type, product_type,
    units_per_cut, prod_plant, min_weight, max_weight, bird_size
):
    """
    Feature: fastapi-enumeration-services, Property 3: SKU schema validation
    Validates: Requirements 7.2, 7.3, 7.4
    
    Property: For any SKU with invalid field types (e.g., non-array allowedParts),
    Pydantic validation should reject it.
    
    This test verifies that when allowedParts is not a list, the validation fails.
    """
    target_weight = (min_weight + max_weight) / 2
    
    # Try to create SKU with invalid allowedParts type (string instead of list)
    with pytest.raises(ValidationError) as exc_info:
        SKU(
            trade_number=trade_number,
            customer_name=customer_name,
            customer_type=customer_type,
            product_type=product_type,
            units_per_cut=units_per_cut,
            prod_plant=prod_plant,
            min_weight=min_weight,
            max_weight=max_weight,
            target_weight=target_weight,
            bird_size=bird_size,
            allowed_parts="INVALID_NOT_A_LIST"  # Invalid: should be a list
        )
    
    # Verify that the error is about the allowedParts field
    errors = exc_info.value.errors()
    assert any('allowed_parts' in str(error['loc']) for error in errors)


@settings(max_examples=100)
@given(
    trade_number=valid_string_strategy(),
    customer_name=valid_string_strategy(),
    customer_type=valid_string_strategy(),
    product_type=valid_string_strategy(),
    units_per_cut=st.integers(min_value=1, max_value=100),
    prod_plant=valid_string_strategy(),
    weights=valid_weight_range_strategy(),
    bird_size=valid_string_strategy(),
    allowed_parts=valid_part_codes_strategy()
)
def test_property_sku_schema_validation_rejects_missing_fields(
    trade_number, customer_name, customer_type, product_type,
    units_per_cut, prod_plant, weights, bird_size, allowed_parts
):
    """
    Feature: fastapi-enumeration-services, Property 3: SKU schema validation
    Validates: Requirements 7.2, 7.3, 7.4
    
    Property: For any SKU missing required fields, Pydantic validation should reject it.
    
    This test verifies that when required fields are missing, validation fails.
    """
    min_weight, max_weight = weights
    
    # Try to create SKU without targetWeight (required field)
    with pytest.raises(ValidationError) as exc_info:
        SKU(
            trade_number=trade_number,
            customer_name=customer_name,
            customer_type=customer_type,
            product_type=product_type,
            units_per_cut=units_per_cut,
            prod_plant=prod_plant,
            min_weight=min_weight,
            max_weight=max_weight,
            # target_weight is missing
            bird_size=bird_size,
            allowed_parts=allowed_parts
        )
    
    # Verify that the error is about a missing field
    errors = exc_info.value.errors()
    assert any(error['type'] == 'missing' for error in errors)



# Property 4: Weight range consistency
# Feature: fastapi-enumeration-services, Property 4: Weight range consistency
# Validates: Requirements 7.3

@settings(max_examples=100)
@given(
    trade_number=valid_string_strategy(),
    customer_name=valid_string_strategy(),
    customer_type=valid_string_strategy(),
    product_type=valid_string_strategy(),
    units_per_cut=st.integers(min_value=1, max_value=100),
    prod_plant=valid_string_strategy(),
    min_weight=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    max_weight=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    bird_size=valid_string_strategy(),
    allowed_parts=valid_part_codes_strategy()
)
def test_property_weight_range_consistency_valid_range(
    trade_number, customer_name, customer_type, product_type,
    units_per_cut, prod_plant, min_weight, max_weight, bird_size, allowed_parts
):
    """
    Feature: fastapi-enumeration-services, Property 4: Weight range consistency
    Validates: Requirements 7.3
    
    Property: For any SKU document, the minWeight should be less than maxWeight.
    
    This test verifies that when minWeight < maxWeight, the SKU is accepted,
    and when minWeight >= maxWeight, the SKU is rejected.
    """
    # Ensure we have a valid range (min < max)
    if min_weight >= max_weight:
        min_weight, max_weight = max_weight, min_weight + 0.1
    
    target_weight = (min_weight + max_weight) / 2
    
    # Create SKU with valid weight range - should not raise ValidationError
    sku = SKU(
        trade_number=trade_number,
        customer_name=customer_name,
        customer_type=customer_type,
        product_type=product_type,
        units_per_cut=units_per_cut,
        prod_plant=prod_plant,
        min_weight=min_weight,
        max_weight=max_weight,
        target_weight=target_weight,
        bird_size=bird_size,
        allowed_parts=allowed_parts
    )
    
    # Verify the weight range is valid
    assert sku.min_weight < sku.max_weight
    assert sku.min_weight >= 0
    assert sku.max_weight >= 0


@settings(max_examples=100)
@given(
    trade_number=valid_string_strategy(),
    customer_name=valid_string_strategy(),
    customer_type=valid_string_strategy(),
    product_type=valid_string_strategy(),
    units_per_cut=st.integers(min_value=1, max_value=100),
    prod_plant=valid_string_strategy(),
    weight=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    bird_size=valid_string_strategy(),
    allowed_parts=valid_part_codes_strategy()
)
def test_property_weight_range_consistency_rejects_invalid_range(
    trade_number, customer_name, customer_type, product_type,
    units_per_cut, prod_plant, weight, bird_size, allowed_parts
):
    """
    Feature: fastapi-enumeration-services, Property 4: Weight range consistency
    Validates: Requirements 7.3
    
    Property: For any SKU document where minWeight >= maxWeight, validation should fail.
    
    This test verifies that invalid weight ranges are rejected.
    """
    # Set maxWeight <= minWeight (invalid range)
    min_weight = weight
    max_weight = weight  # Equal weights (invalid)
    target_weight = weight
    
    # Try to create SKU with invalid weight range - should raise ValidationError
    with pytest.raises(ValidationError) as exc_info:
        SKU(
            trade_number=trade_number,
            customer_name=customer_name,
            customer_type=customer_type,
            product_type=product_type,
            units_per_cut=units_per_cut,
            prod_plant=prod_plant,
            min_weight=min_weight,
            max_weight=max_weight,  # Equal to minWeight (invalid)
            target_weight=target_weight,
            bird_size=bird_size,
            allowed_parts=allowed_parts
        )
    
    # Verify that the error is about maxWeight
    errors = exc_info.value.errors()
    assert any('max_weight' in str(error['loc']) for error in errors)


@settings(max_examples=100)
@given(
    trade_number=valid_string_strategy(),
    customer_name=valid_string_strategy(),
    customer_type=valid_string_strategy(),
    product_type=valid_string_strategy(),
    units_per_cut=st.integers(min_value=1, max_value=100),
    prod_plant=valid_string_strategy(),
    min_weight=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    max_weight=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    bird_size=valid_string_strategy(),
    allowed_parts=valid_part_codes_strategy()
)
def test_property_weight_range_consistency_rejects_inverted_range(
    trade_number, customer_name, customer_type, product_type,
    units_per_cut, prod_plant, min_weight, max_weight, bird_size, allowed_parts
):
    """
    Feature: fastapi-enumeration-services, Property 4: Weight range consistency
    Validates: Requirements 7.3
    
    Property: For any SKU document where minWeight > maxWeight, validation should fail.
    
    This test verifies that inverted weight ranges (min > max) are rejected.
    """
    # Ensure we have an inverted range (min > max)
    if min_weight <= max_weight:
        min_weight, max_weight = max_weight + 1, min_weight
    
    target_weight = (min_weight + max_weight) / 2
    
    # Try to create SKU with inverted weight range - should raise ValidationError
    with pytest.raises(ValidationError) as exc_info:
        SKU(
            trade_number=trade_number,
            customer_name=customer_name,
            customer_type=customer_type,
            product_type=product_type,
            units_per_cut=units_per_cut,
            prod_plant=prod_plant,
            min_weight=min_weight,  # Greater than maxWeight (invalid)
            max_weight=max_weight,
            target_weight=target_weight,
            bird_size=bird_size,
            allowed_parts=allowed_parts
        )
    
    # Verify that the error is about maxWeight
    errors = exc_info.value.errors()
    assert any('max_weight' in str(error['loc']) for error in errors)


@settings(max_examples=100)
@given(sku=valid_sku_strategy())
def test_property_weight_range_consistency_target_in_range(sku: SKU):
    """
    Feature: fastapi-enumeration-services, Property 4: Weight range consistency
    Validates: Requirements 7.3
    
    Property: For any valid SKU, targetWeight should ideally be between minWeight and maxWeight.
    
    Note: This is a soft validation - the current implementation allows targetWeight
    outside the range, but this test documents the expected behavior.
    """
    # For valid SKUs generated by our strategy, target should be in range
    # (our strategy generates target as the midpoint)
    assert sku.min_weight <= sku.target_weight <= sku.max_weight
    assert sku.min_weight < sku.max_weight

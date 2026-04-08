"""
Property-based tests for Configuration models.

This module contains property-based tests using Hypothesis to verify that the Config
models behave correctly across a wide range of inputs. Each test runs 100+ iterations
with randomly generated data to ensure comprehensive coverage.

Tests are tagged with the feature name and property number from the design document.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from pydantic import ValidationError
from datetime import datetime, timezone
from typing import Union

from models.config import Config, ConfigUpdate, ValueType, BatchConfigUpdate


# Custom strategies for generating valid configuration data

def valid_key_strategy():
    """Generate valid configuration keys."""
    return st.text(
        alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='.-_'
        ),
        min_size=1,
        max_size=200
    ).filter(lambda s: s.strip() and not s.startswith('.') and not s.endswith('.'))


def valid_description_strategy():
    """Generate valid descriptions."""
    return st.text(
        alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd', 'P', 'Z'),
        ),
        min_size=1,
        max_size=500
    ).filter(lambda s: s.strip())


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


def mismatched_value_for_type_strategy(value_type: ValueType) -> st.SearchStrategy:
    """Generate a value that does NOT match the specified ValueType."""
    if value_type == ValueType.INT:
        # Return string, float, or bool (but not int)
        # Ensure strings are not numeric-looking
        return st.one_of(
            st.text(min_size=1, max_size=50).filter(lambda s: not s.isdigit() and s.strip()),
            st.floats(allow_nan=False, allow_infinity=False).filter(lambda x: x != int(x) if not isinstance(x, bool) else True)
        )
    elif value_type == ValueType.STRING:
        # Return int, float, or bool (but not string)
        return st.one_of(
            st.integers().filter(lambda x: not isinstance(x, bool)),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans()
        )
    elif value_type == ValueType.FLOAT:
        # Return string or bool (but not int or float)
        return st.one_of(
            st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
            st.booleans()
        )
    elif value_type == ValueType.BOOL:
        # Return int, string, or float (but not bool)
        return st.one_of(
            st.integers().filter(lambda x: not isinstance(x, bool)),
            st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
            st.floats(allow_nan=False, allow_infinity=False)
        )
    else:
        raise ValueError(f"Unknown ValueType: {value_type}")


# Property 7: Value type validation
# Feature: fastapi-enumeration-services, Property 7: Value type validation
# Validates: Requirements 11.5

@settings(max_examples=100)
@given(
    key=valid_key_strategy(),
    value_type=st.sampled_from(list(ValueType)),
    description=valid_description_strategy(),
    data=st.data()
)
def test_property_value_type_validation_accepts_matching_types(
    key: str,
    value_type: ValueType,
    description: str,
    data
):
    """
    **Validates: Requirements 11.5**
    
    Feature: fastapi-enumeration-services, Property 7: Value type validation
    
    Property: For any configuration document, the value field should match the type
    specified by valueType (int values for "int", string values for "string",
    float values for "float", bool values for "bool").
    
    This test verifies that:
    1. Config models accept values that match their declared valueType
    2. Type validation works correctly for all supported types
    3. The validation happens at model creation time
    """
    # Generate a value that matches the value_type
    value = data.draw(value_for_type_strategy(value_type))
    
    # Create a Config with matching value and valueType
    config = Config(
        key=key,
        value=value,
        value_type=value_type,
        description=description,
        updated_at=datetime.now(timezone.utc)
    )
    
    # Verify the config was created successfully
    assert config.key == key
    assert config.value == value
    assert config.value_type == value_type
    assert config.description == description
    
    # Verify the value type matches expectations
    if value_type == ValueType.INT:
        assert isinstance(config.value, int) and not isinstance(config.value, bool)
    elif value_type == ValueType.STRING:
        assert isinstance(config.value, str)
    elif value_type == ValueType.FLOAT:
        assert isinstance(config.value, (int, float)) and not isinstance(config.value, bool)
    elif value_type == ValueType.BOOL:
        assert isinstance(config.value, bool)


@settings(max_examples=100)
@given(
    key=valid_key_strategy(),
    value_type=st.sampled_from(list(ValueType)),
    description=valid_description_strategy(),
    data=st.data()
)
def test_property_value_type_validation_rejects_mismatched_types(
    key: str,
    value_type: ValueType,
    description: str,
    data
):
    """
    **Validates: Requirements 11.5**
    
    Feature: fastapi-enumeration-services, Property 7: Value type validation
    
    Property: For any configuration document, if the value field does not match
    the type specified by valueType, validation should fail with a clear error.
    
    This test verifies that:
    1. Config models reject values that don't match their declared valueType
    2. Validation errors are raised for type mismatches
    3. The error messages are clear and helpful
    """
    # Generate a value that does NOT match the value_type
    mismatched_value = data.draw(mismatched_value_for_type_strategy(value_type))
    
    # Attempt to create a Config with mismatched value and valueType
    with pytest.raises(ValidationError) as exc_info:
        Config(
            key=key,
            value=mismatched_value,
            value_type=value_type,
            description=description,
            updated_at=datetime.now(timezone.utc)
        )
    
    # Verify that the error is about value type mismatch
    error_message = str(exc_info.value)
    assert "Value must be of type" in error_message or "type" in error_message.lower()


@settings(max_examples=100)
@given(
    key=valid_key_strategy(),
    value_type=st.sampled_from(list(ValueType)),
    description=valid_description_strategy(),
    data=st.data()
)
def test_property_config_update_value_type_validation(
    key: str,
    value_type: ValueType,
    description: str,
    data
):
    """
    **Validates: Requirements 11.5**
    
    Feature: fastapi-enumeration-services, Property 7: Value type validation
    
    Property: ConfigUpdate models should also validate that values match their
    declared valueType, ensuring consistency across create and update operations.
    
    This test verifies that:
    1. ConfigUpdate models validate value types correctly
    2. Both matching and mismatched types are handled appropriately
    """
    # Test with matching value type
    matching_value = data.draw(value_for_type_strategy(value_type))
    
    config_update = ConfigUpdate(
        value=matching_value,
        value_type=value_type,
        description=description
    )
    
    assert config_update.value == matching_value
    assert config_update.value_type == value_type
    
    # Test with mismatched value type
    mismatched_value = data.draw(mismatched_value_for_type_strategy(value_type))
    
    with pytest.raises(ValidationError) as exc_info:
        ConfigUpdate(
            value=mismatched_value,
            value_type=value_type,
            description=description
        )
    
    error_message = str(exc_info.value)
    assert "Value must be of type" in error_message or "type" in error_message.lower()


@settings(max_examples=100)
@given(
    key=valid_key_strategy(),
    description=valid_description_strategy()
)
def test_property_int_type_rejects_bool(key: str, description: str):
    """
    **Validates: Requirements 11.5**
    
    Feature: fastapi-enumeration-services, Property 7: Value type validation
    
    Property: When valueType is INT, boolean values should be rejected even though
    bool is a subclass of int in Python. This ensures strict type checking.
    
    This test verifies the special case handling for bool vs int.
    """
    # Attempt to create a Config with bool value and INT type
    with pytest.raises(ValidationError) as exc_info:
        Config(
            key=key,
            value=True,  # Boolean value
            value_type=ValueType.INT,  # INT type
            description=description,
            updated_at=datetime.now(timezone.utc)
        )
    
    # Verify the error mentions bool or type mismatch
    error_message = str(exc_info.value)
    assert "bool" in error_message.lower() or "type" in error_message.lower()


@settings(max_examples=100)
@given(
    key=valid_key_strategy(),
    description=valid_description_strategy(),
    int_value=st.integers(min_value=-1000000, max_value=1000000)
)
def test_property_float_type_accepts_int(key: str, description: str, int_value: int):
    """
    **Validates: Requirements 11.5**
    
    Feature: fastapi-enumeration-services, Property 7: Value type validation
    
    Property: When valueType is FLOAT, integer values should be accepted since
    integers can be represented as floats without loss of precision.
    
    This test verifies that FLOAT type is flexible enough to accept int values.
    """
    # Create a Config with int value and FLOAT type (should succeed)
    config = Config(
        key=key,
        value=int_value,
        value_type=ValueType.FLOAT,
        description=description,
        updated_at=datetime.now(timezone.utc)
    )
    
    assert config.value == int_value
    assert config.value_type == ValueType.FLOAT
    assert isinstance(config.value, (int, float))



# Property 8: Numeric range validation
# Feature: fastapi-enumeration-services, Property 8: Numeric range validation
# Validates: Requirements 11.6

@settings(max_examples=100)
@given(
    key=valid_key_strategy(),
    value_type=st.sampled_from([ValueType.INT, ValueType.FLOAT]),
    description=valid_description_strategy(),
    data=st.data()
)
def test_property_numeric_range_validation_accepts_values_within_range(
    key: str,
    value_type: ValueType,
    description: str,
    data
):
    """
    **Validates: Requirements 11.6**
    
    Feature: fastapi-enumeration-services, Property 8: Numeric range validation
    
    Property: For any configuration document with numeric value and minValue/maxValue
    constraints, the value should be greater than or equal to minValue (if present)
    and less than or equal to maxValue (if present).
    
    This test verifies that:
    1. Config models accept numeric values within the specified range
    2. Range validation works for both int and float types
    3. Values at the boundaries (min and max) are accepted
    """
    # Generate min and max values
    if value_type == ValueType.INT:
        # For INT type, use integer boundaries
        min_value = float(data.draw(st.integers(min_value=-1000, max_value=999)))
        max_value = float(data.draw(st.integers(min_value=int(min_value) + 1, max_value=1000)))
        value = data.draw(st.integers(min_value=int(min_value), max_value=int(max_value)))
    else:  # FLOAT
        # For FLOAT type, use float boundaries
        min_value = data.draw(st.floats(min_value=-1000.0, max_value=999.0, allow_nan=False, allow_infinity=False))
        max_value = data.draw(st.floats(min_value=min_value + 0.1, max_value=1000.0, allow_nan=False, allow_infinity=False))
        value = data.draw(st.floats(min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False))
    
    # Create a Config with value within range
    config = Config(
        key=key,
        value=value,
        value_type=value_type,
        description=description,
        updated_at=datetime.now(timezone.utc),
        min_value=min_value,
        max_value=max_value
    )
    
    # Verify the config was created successfully
    assert config.value == value
    assert config.min_value == min_value
    assert config.max_value == max_value
    assert min_value <= config.value <= max_value


@settings(max_examples=100)
@given(
    key=valid_key_strategy(),
    value_type=st.sampled_from([ValueType.INT, ValueType.FLOAT]),
    description=valid_description_strategy(),
    data=st.data()
)
def test_property_numeric_range_validation_rejects_values_below_minimum(
    key: str,
    value_type: ValueType,
    description: str,
    data
):
    """
    **Validates: Requirements 11.6**
    
    Feature: fastapi-enumeration-services, Property 8: Numeric range validation
    
    Property: For any configuration document with numeric value and minValue constraint,
    values below minValue should be rejected with a clear error.
    
    This test verifies that:
    1. Config models reject numeric values below minValue
    2. Validation errors are raised for out-of-range values
    3. The error messages are clear and helpful
    """
    # Generate min value and a value below it
    min_value = data.draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    
    if value_type == ValueType.INT:
        value = data.draw(st.integers(max_value=int(min_value) - 1))
    else:  # FLOAT
        value = data.draw(st.floats(max_value=min_value - 0.1, allow_nan=False, allow_infinity=False))
    
    # Attempt to create a Config with value below minimum
    with pytest.raises(ValidationError) as exc_info:
        Config(
            key=key,
            value=value,
            value_type=value_type,
            description=description,
            updated_at=datetime.now(timezone.utc),
            min_value=min_value
        )
    
    # Verify that the error is about value being below minimum
    error_message = str(exc_info.value)
    assert "below minimum" in error_message.lower() or "minimum" in error_message.lower()


@settings(max_examples=100)
@given(
    key=valid_key_strategy(),
    value_type=st.sampled_from([ValueType.INT, ValueType.FLOAT]),
    description=valid_description_strategy(),
    data=st.data()
)
def test_property_numeric_range_validation_rejects_values_above_maximum(
    key: str,
    value_type: ValueType,
    description: str,
    data
):
    """
    **Validates: Requirements 11.6**
    
    Feature: fastapi-enumeration-services, Property 8: Numeric range validation
    
    Property: For any configuration document with numeric value and maxValue constraint,
    values above maxValue should be rejected with a clear error.
    
    This test verifies that:
    1. Config models reject numeric values above maxValue
    2. Validation errors are raised for out-of-range values
    3. The error messages are clear and helpful
    """
    # Generate max value and a value above it
    max_value = data.draw(st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    
    if value_type == ValueType.INT:
        value = data.draw(st.integers(min_value=int(max_value) + 1))
    else:  # FLOAT
        value = data.draw(st.floats(min_value=max_value + 0.1, allow_nan=False, allow_infinity=False))
    
    # Attempt to create a Config with value above maximum
    with pytest.raises(ValidationError) as exc_info:
        Config(
            key=key,
            value=value,
            value_type=value_type,
            description=description,
            updated_at=datetime.now(timezone.utc),
            max_value=max_value
        )
    
    # Verify that the error is about value exceeding maximum
    error_message = str(exc_info.value)
    assert "exceed" in error_message.lower() or "maximum" in error_message.lower()


@settings(max_examples=100)
@given(
    key=valid_key_strategy(),
    description=valid_description_strategy(),
    data=st.data()
)
def test_property_numeric_range_validation_ignores_non_numeric_types(
    key: str,
    description: str,
    data
):
    """
    **Validates: Requirements 11.6**
    
    Feature: fastapi-enumeration-services, Property 8: Numeric range validation
    
    Property: For any configuration document with non-numeric value types (string, bool),
    minValue and maxValue constraints should be ignored (not cause errors).
    
    This test verifies that:
    1. String and bool values are not validated against numeric ranges
    2. minValue and maxValue can be present but are ignored for non-numeric types
    """
    # Test with STRING type
    string_value = data.draw(st.text(min_size=1, max_size=100))
    
    config_string = Config(
        key=key,
        value=string_value,
        value_type=ValueType.STRING,
        description=description,
        updated_at=datetime.now(timezone.utc),
        min_value=0.0,
        max_value=100.0
    )
    
    assert config_string.value == string_value
    assert config_string.value_type == ValueType.STRING
    
    # Test with BOOL type
    bool_value = data.draw(st.booleans())
    
    config_bool = Config(
        key=key + "_bool",
        value=bool_value,
        value_type=ValueType.BOOL,
        description=description,
        updated_at=datetime.now(timezone.utc),
        min_value=0.0,
        max_value=100.0
    )
    
    assert config_bool.value == bool_value
    assert config_bool.value_type == ValueType.BOOL


@settings(max_examples=100)
@given(
    key=valid_key_strategy(),
    value_type=st.sampled_from([ValueType.INT, ValueType.FLOAT]),
    description=valid_description_strategy(),
    data=st.data()
)
def test_property_config_update_numeric_range_validation(
    key: str,
    value_type: ValueType,
    description: str,
    data
):
    """
    **Validates: Requirements 11.6**
    
    Feature: fastapi-enumeration-services, Property 8: Numeric range validation
    
    Property: ConfigUpdate models should also validate numeric ranges, ensuring
    consistency across create and update operations.
    
    This test verifies that:
    1. ConfigUpdate models validate numeric ranges correctly
    2. Both in-range and out-of-range values are handled appropriately
    """
    # Generate min and max values
    if value_type == ValueType.INT:
        min_value = float(data.draw(st.integers(min_value=-1000, max_value=999)))
        max_value = float(data.draw(st.integers(min_value=int(min_value) + 1, max_value=1000)))
        in_range_value = data.draw(st.integers(min_value=int(min_value), max_value=int(max_value)))
    else:
        min_value = data.draw(st.floats(min_value=-1000.0, max_value=999.0, allow_nan=False, allow_infinity=False))
        max_value = data.draw(st.floats(min_value=min_value + 0.1, max_value=1000.0, allow_nan=False, allow_infinity=False))
        in_range_value = data.draw(st.floats(min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False))
    
    config_update = ConfigUpdate(
        value=in_range_value,
        value_type=value_type,
        description=description,
        min_value=min_value,
        max_value=max_value
    )
    
    assert config_update.value == in_range_value
    assert min_value <= config_update.value <= max_value
    
    # Test with value below minimum
    if value_type == ValueType.INT:
        below_min_value = int(min_value) - 1
    else:
        below_min_value = min_value - 0.1
    
    with pytest.raises(ValidationError) as exc_info:
        ConfigUpdate(
            value=below_min_value,
            value_type=value_type,
            description=description,
            min_value=min_value,
            max_value=max_value
        )
    
    error_message = str(exc_info.value)
    assert "below minimum" in error_message.lower() or "minimum" in error_message.lower()



# Property 10: Config schema validation
# Feature: fastapi-enumeration-services, Property 10: Config schema validation
# Validates: Requirements 13.1, 13.2, 13.3, 13.5

@settings(max_examples=100)
@given(
    key=valid_key_strategy(),
    value_type=st.sampled_from(list(ValueType)),
    description=valid_description_strategy(),
    data=st.data()
)
def test_property_config_schema_validation_all_required_fields(
    key: str,
    value_type: ValueType,
    description: str,
    data
):
    """
    **Validates: Requirements 13.1, 13.2, 13.3, 13.5**
    
    Feature: fastapi-enumeration-services, Property 10: Config schema validation
    
    Property: For any configuration document, it should contain all required fields
    (key, value, valueType, description, updatedAt), with _id equal to key, valueType
    being one of the allowed values ("int", "string", "float", "bool"), and updatedAt
    being a valid ISO 8601 formatted string.
    
    This test verifies that:
    1. Config models require all mandatory fields
    2. All fields are correctly populated
    3. Field types are validated
    """
    # Generate a value matching the value_type
    value = data.draw(value_for_type_strategy(value_type))
    updated_at = datetime.now(timezone.utc)
    
    # Create a Config with all required fields
    config = Config(
        key=key,
        value=value,
        value_type=value_type,
        description=description,
        updated_at=updated_at
    )
    
    # Verify all required fields are present
    assert config.key == key
    assert config.value == value
    assert config.value_type == value_type
    assert config.description == description
    assert config.updated_at == updated_at
    
    # Verify valueType is one of the allowed values
    assert config.value_type in [ValueType.INT, ValueType.STRING, ValueType.FLOAT, ValueType.BOOL]
    
    # Verify updatedAt is a datetime object (will be serialized to ISO 8601 string)
    assert isinstance(config.updated_at, datetime)


@settings(max_examples=100)
@given(
    value_type=st.sampled_from(list(ValueType)),
    description=valid_description_strategy(),
    data=st.data()
)
def test_property_config_schema_validation_missing_required_fields(
    value_type: ValueType,
    description: str,
    data
):
    """
    **Validates: Requirements 13.1, 13.2, 13.3, 13.5**
    
    Feature: fastapi-enumeration-services, Property 10: Config schema validation
    
    Property: For any configuration document missing required fields, validation
    should fail with a clear error indicating which fields are missing.
    
    This test verifies that:
    1. Config models reject documents missing required fields
    2. Validation errors are raised for missing fields
    """
    value = data.draw(value_for_type_strategy(value_type))
    
    # Test missing key
    with pytest.raises(ValidationError):
        Config(
            value=value,
            value_type=value_type,
            description=description,
            updated_at=datetime.now(timezone.utc)
        )
    
    # Test missing value
    with pytest.raises(ValidationError):
        Config(
            key="test_key",
            value_type=value_type,
            description=description,
            updated_at=datetime.now(timezone.utc)
        )
    
    # Test missing value_type
    with pytest.raises(ValidationError):
        Config(
            key="test_key",
            value=value,
            description=description,
            updated_at=datetime.now(timezone.utc)
        )
    
    # Test missing description
    with pytest.raises(ValidationError):
        Config(
            key="test_key",
            value=value,
            value_type=value_type,
            updated_at=datetime.now(timezone.utc)
        )
    
    # Test missing updated_at
    with pytest.raises(ValidationError):
        Config(
            key="test_key",
            value=value,
            value_type=value_type,
            description=description
        )


@settings(max_examples=100)
@given(
    key=valid_key_strategy(),
    value_type=st.sampled_from(list(ValueType)),
    description=valid_description_strategy(),
    data=st.data()
)
def test_property_config_schema_validation_optional_fields(
    key: str,
    value_type: ValueType,
    description: str,
    data
):
    """
    **Validates: Requirements 13.1, 13.2, 13.3, 13.5**
    
    Feature: fastapi-enumeration-services, Property 10: Config schema validation
    
    Property: For any configuration document, minValue and maxValue fields are optional
    and should be accepted when present or absent.
    
    This test verifies that:
    1. Config models accept documents without optional fields
    2. Config models accept documents with optional fields
    3. Optional fields are correctly populated when provided
    """
    value = data.draw(value_for_type_strategy(value_type))
    
    # Test without optional fields
    config_without_optional = Config(
        key=key,
        value=value,
        value_type=value_type,
        description=description,
        updated_at=datetime.now(timezone.utc)
    )
    
    assert config_without_optional.min_value is None
    assert config_without_optional.max_value is None
    
    # Test with optional fields (only for numeric types)
    if value_type in [ValueType.INT, ValueType.FLOAT]:
        if value_type == ValueType.INT:
            min_value = float(data.draw(st.integers(min_value=-1000, max_value=999)))
            max_value = float(data.draw(st.integers(min_value=int(min_value) + 1, max_value=1000)))
            value_in_range = data.draw(st.integers(min_value=int(min_value), max_value=int(max_value)))
        else:
            min_value = data.draw(st.floats(min_value=-1000.0, max_value=999.0, allow_nan=False, allow_infinity=False))
            max_value = data.draw(st.floats(min_value=min_value + 0.1, max_value=1000.0, allow_nan=False, allow_infinity=False))
            value_in_range = data.draw(st.floats(min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False))
        
        config_with_optional = Config(
            key=key + "_with_optional",
            value=value_in_range,
            value_type=value_type,
            description=description,
            updated_at=datetime.now(timezone.utc),
            min_value=min_value,
            max_value=max_value
        )
        
        assert config_with_optional.min_value == min_value
        assert config_with_optional.max_value == max_value


@settings(max_examples=100)
@given(
    key=valid_key_strategy(),
    value_type=st.sampled_from(list(ValueType)),
    description=valid_description_strategy(),
    data=st.data()
)
def test_property_config_schema_validation_field_constraints(
    key: str,
    value_type: ValueType,
    description: str,
    data
):
    """
    **Validates: Requirements 13.1, 13.2, 13.3, 13.5**
    
    Feature: fastapi-enumeration-services, Property 10: Config schema validation
    
    Property: For any configuration document, field constraints should be enforced
    (e.g., key and description must be non-empty strings with length limits).
    
    This test verifies that:
    1. Config models enforce field length constraints
    2. Empty strings are rejected for required string fields
    """
    value = data.draw(value_for_type_strategy(value_type))
    
    # Test empty key (should fail)
    with pytest.raises(ValidationError):
        Config(
            key="",
            value=value,
            value_type=value_type,
            description=description,
            updated_at=datetime.now(timezone.utc)
        )
    
    # Test empty description (should fail)
    with pytest.raises(ValidationError):
        Config(
            key=key,
            value=value,
            value_type=value_type,
            description="",
            updated_at=datetime.now(timezone.utc)
        )
    
    # Test key that's too long (should fail)
    with pytest.raises(ValidationError):
        Config(
            key="a" * 201,  # Max length is 200
            value=value,
            value_type=value_type,
            description=description,
            updated_at=datetime.now(timezone.utc)
        )
    
    # Test description that's too long (should fail)
    with pytest.raises(ValidationError):
        Config(
            key=key,
            value=value,
            value_type=value_type,
            description="a" * 501,  # Max length is 500
            updated_at=datetime.now(timezone.utc)
        )

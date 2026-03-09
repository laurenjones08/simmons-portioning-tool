# Design Document

## Overview

This document describes the design for a microservices-based enumeration engine system consisting of two FastAPI services and a MongoDB database. The system provides SKU (Stock Keeping Unit) management and centralized configuration management capabilities.

The architecture follows modern microservices principles with:
- **Service isolation**: Each API service runs independently in its own container
- **Layered architecture**: Repository, service, and router layers for separation of concerns
- **Dependency injection**: Services are injected into routers for testability
- **Schema validation**: Pydantic models ensure data integrity
- **Containerization**: Docker and docker-compose for consistent deployment

### System Components

1. **Enumeration API**: Manages SKU data with search and retrieval capabilities
2. **Global Config API**: Manages system-wide configuration key-value pairs
3. **MongoDB**: Persistent storage for both services

## Architecture

### High-Level Architecture

```mermaid
graph TB
    Client[Client Applications]
    
    subgraph Docker Environment
        EnumAPI[Enumeration API<br/>:8000]
        ConfigAPI[Global Config API<br/>:8001]
        MongoDB[(MongoDB<br/>:27017)]
    end
    
    Client -->|HTTP| EnumAPI
    Client -->|HTTP| ConfigAPI
    EnumAPI -->|PyMongo| MongoDB
    ConfigAPI -->|PyMongo| MongoDB
```

### Service Architecture Pattern

Both APIs follow the same three-layer architecture:

```mermaid
graph TB
    Router[Router Layer<br/>FastAPI Endpoints]
    Service[Service Layer<br/>Business Logic]
    Repository[Repository Layer<br/>Data Access]
    DB[(MongoDB)]
    
    Router -->|Dependency Injection| Service
    Service --> Repository
    Repository --> DB
```

**Layer Responsibilities:**

- **Router Layer**: Handles HTTP requests/responses, request validation, and response serialization
- **Service Layer**: Contains business logic, orchestrates repository calls, and handles error cases
- **Repository Layer**: Provides data access abstraction, executes MongoDB queries, and maps documents to models

### Docker Architecture

```mermaid
graph LR
    subgraph Docker Compose
        subgraph Enumeration Container
            EnumApp[FastAPI App<br/>Uvicorn Server]
        end
        
        subgraph Config Container
            ConfigApp[FastAPI App<br/>Uvicorn Server]
        end
        
        subgraph MongoDB Container
            MongoData[(MongoDB<br/>Data Volume)]
        end
    end
    
    EnumApp -.->|mongo_net| MongoData
    ConfigApp -.->|mongo_net| MongoData
```

**Docker Components:**
- Each service has its own Dockerfile based on Python 3.11
- Services communicate over a Docker network named `mongo_net`
- MongoDB data persists in a named volume `mongo_data`
- Environment variables configure MongoDB connection strings

## Components and Interfaces

### Enumeration API Components

#### 1. Router Layer (`routers/sku_router.py`)

**Endpoints:**

```python
GET /health
# Returns: {"status": "healthy"}
# Purpose: Health check for monitoring

GET /metrics
# Returns: Prometheus-formatted metrics
# Purpose: Expose metrics for monitoring systems

GET /skus/{trade_number}
# Parameters: trade_number (path parameter, string)
# Returns: SKU document or 404
# Purpose: Retrieve single SKU by trade number

POST /skus/search
# Body: SearchCriteria (JSON)
# Returns: List of SKU documents
# Purpose: Search SKUs by filter criteria

POST /skus/batch
# Body: Array of SKU documents (JSON)
# Returns: BatchImportResult with success/failure counts
# Purpose: Bulk import SKUs

GET /skus/export
# Returns: Array of all SKU documents (JSON)
# Purpose: Bulk export all SKUs
```

**Dependencies:**
- Injects `SKUService` via FastAPI dependency injection

#### 2. Service Layer (`services/sku_service.py`)

**Class: SKUService**

```python
class SKUService:
    """Business logic for SKU operations."""
    
    def __init__(self, repository: SKURepository):
        """Initialize service with repository dependency."""
        
    def get_sku_by_trade_number(self, trade_number: str) -> Optional[SKU]:
        """Retrieve SKU by trade number.
        
        Args:
            trade_number: The unique trade number identifier
            
        Returns:
            SKU model if found, None otherwise
        """
        
    def search_skus(self, criteria: Dict[str, Any]) -> List[SKU]:
        """Search SKUs by filter criteria.
        
        Args:
            criteria: Dictionary of field filters
            
        Returns:
            List of matching SKU models
        """
    
    def batch_import(self, skus: List[SKU], validate_only: bool = False) -> BatchImportResult:
        """Import multiple SKUs with validation.
        
        Args:
            skus: List of SKU models to import
            validate_only: If True, only validate without inserting
            
        Returns:
            BatchImportResult with success/failure counts
        """
    
    def export_all(self, filter_criteria: Optional[Dict] = None) -> List[SKU]:
        """Export all SKUs matching criteria.
        
        Args:
            filter_criteria: Optional MongoDB filter
            
        Returns:
            List of all matching SKU models
        """
```

#### 3. Repository Layer (`repositories/sku_repository.py`)

**Class: SKURepository**

```python
class SKURepository:
    """Data access layer for SKU collection."""
    
    def __init__(self, database: Database):
        """Initialize repository with MongoDB database."""
        
    def find_by_trade_number(self, trade_number: str) -> Optional[Dict]:
        """Find SKU document by trade number (_id).
        
        Args:
            trade_number: The unique trade number identifier
            
        Returns:
            Document dict if found, None otherwise
        """
        
    def find_by_criteria(self, criteria: Dict[str, Any]) -> List[Dict]:
        """Find SKU documents matching criteria.
        
        Args:
            criteria: MongoDB query filter
            
        Returns:
            List of matching documents
        """
```

#### 4. Models (`models/sku.py`)

**Pydantic Models:**

```python
class SKU(BaseModel):
    """SKU data model with validation."""
    
    trade_number: str = Field(..., description="Unique trade number identifier")
    customer_name: str = Field(..., description="Customer name")
    customer_type: str = Field(..., description="Customer type code")
    product_type: str = Field(..., description="Product type")
    units_per_cut: int = Field(..., ge=1, description="Units per cut")
    prod_plant: str = Field(..., description="Production plant code")
    min_weight: float = Field(..., ge=0, description="Minimum weight in grams")
    max_weight: float = Field(..., ge=0, description="Maximum weight in grams")
    target_weight: float = Field(..., ge=0, description="Target weight in grams")
    bird_size: str = Field(..., description="Bird size code")
    allowed_parts: List[str] = Field(..., description="List of allowed part codes")
    
    @validator('max_weight')
    def validate_weight_range(cls, v, values):
        """Ensure max_weight > min_weight."""
        if 'min_weight' in values and v <= values['min_weight']:
            raise ValueError('max_weight must be greater than min_weight')
        return v

class SearchCriteria(BaseModel):
    """Search filter criteria."""
    
    customer_type: Optional[str] = None
    product_type: Optional[str] = None
    prod_plant: Optional[str] = None
    bird_size: Optional[str] = None
    # Additional filter fields as needed
```

#### 5. Configuration (`config.py`)

```python
class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    mongodb_url: str = Field(..., env="MONGODB_URL")
    mongodb_database: str = Field(default="enumeration_db", env="MONGODB_DATABASE")
    
    class Config:
        env_file = ".env"
```

#### 6. Database Connection (`database.py`)

```python
def get_database() -> Database:
    """Get MongoDB database connection.
    
    Returns:
        MongoDB Database instance
        
    Usage:
        Used as FastAPI dependency for repository initialization
    """
```

### Global Config API Components

#### 1. Router Layer (`routers/config_router.py`)

**Endpoints:**

```python
GET /health
# Returns: {"status": "healthy"}
# Purpose: Health check for monitoring

GET /metrics
# Returns: Prometheus-formatted metrics
# Purpose: Expose metrics for monitoring systems

GET /config/{key}
# Parameters: key (path parameter, string)
# Returns: Config document or 404
# Purpose: Retrieve single configuration by key

PUT /config/{key}
# Parameters: key (path parameter, string)
# Body: ConfigUpdate (JSON)
# Returns: Updated config document
# Purpose: Create or update configuration

GET /config
# Returns: List of all config documents
# Purpose: Retrieve all configurations

POST /config/batch
# Body: Array of ConfigUpdate objects (JSON)
# Returns: BatchUpdateResult with success/failure counts
# Purpose: Bulk update configurations
```

**Dependencies:**
- Injects `ConfigService` via FastAPI dependency injection

#### 2. Service Layer (`services/config_service.py`)

**Class: ConfigService**

```python
class ConfigService:
    """Business logic for configuration operations."""
    
    def __init__(self, repository: ConfigRepository):
        """Initialize service with repository dependency."""
        
    def get_config_by_key(self, key: str) -> Optional[Config]:
        """Retrieve configuration by key.
        
        Args:
            key: The configuration key identifier
            
        Returns:
            Config model if found, None otherwise
        """
        
    def update_config(self, key: str, update: ConfigUpdate) -> Config:
        """Create or update configuration.
        
        Args:
            key: The configuration key identifier
            update: Configuration update data
            
        Returns:
            Updated Config model
            
        Raises:
            ValueError: If value doesn't match valueType or validation constraints
        """
        
    def get_all_configs(self) -> List[Config]:
        """Retrieve all configurations.
        
        Returns:
            List of all Config models
        """
    
    def batch_update(
        self,
        updates: List[BatchConfigUpdate],
        validate_only: bool = False
    ) -> BatchUpdateResult:
        """Update multiple configurations with validation.
        
        Args:
            updates: List of configuration updates
            validate_only: If True, only validate without updating
            
        Returns:
            BatchUpdateResult with success/failure counts
        """
```

#### 3. Repository Layer (`repositories/config_repository.py`)

**Class: ConfigRepository**

```python
class ConfigRepository:
    """Data access layer for global_config collection."""
    
    def __init__(self, database: Database):
        """Initialize repository with MongoDB database."""
        
    def find_by_key(self, key: str) -> Optional[Dict]:
        """Find config document by key (_id).
        
        Args:
            key: The configuration key identifier
            
        Returns:
            Document dict if found, None otherwise
        """
        
    def upsert(self, key: str, document: Dict) -> Dict:
        """Insert or update config document.
        
        Args:
            key: The configuration key identifier
            document: Configuration document to save
            
        Returns:
            Updated document
        """
        
    def find_all(self) -> List[Dict]:
        """Find all config documents.
        
        Returns:
            List of all configuration documents
        """
```

#### 4. Models (`models/config.py`)

**Pydantic Models:**

```python
class ValueType(str, Enum):
    """Supported configuration value types."""
    INT = "int"
    STRING = "string"
    FLOAT = "float"
    BOOL = "bool"

class Config(BaseModel):
    """Configuration data model."""
    
    key: str = Field(..., description="Configuration key identifier")
    value: Union[int, str, float, bool] = Field(..., description="Configuration value")
    value_type: ValueType = Field(..., description="Type of the value")
    description: str = Field(..., description="Human-readable description")
    updated_at: datetime = Field(..., description="Last update timestamp")
    min_value: Optional[float] = Field(None, description="Minimum value for numeric types")
    max_value: Optional[float] = Field(None, description="Maximum value for numeric types")
    
    @validator('value')
    def validate_value_type(cls, v, values):
        """Ensure value matches declared value_type."""
        if 'value_type' not in values:
            return v
            
        value_type = values['value_type']
        type_map = {
            ValueType.INT: int,
            ValueType.STRING: str,
            ValueType.FLOAT: (int, float),
            ValueType.BOOL: bool
        }
        
        expected_type = type_map[value_type]
        if not isinstance(v, expected_type):
            raise ValueError(f'Value must be of type {value_type}')
        return v
    
    @validator('value')
    def validate_numeric_range(cls, v, values):
        """Ensure numeric values are within min/max constraints."""
        if not isinstance(v, (int, float)):
            return v
            
        min_val = values.get('min_value')
        max_val = values.get('max_value')
        
        if min_val is not None and v < min_val:
            raise ValueError(f'Value {v} is below minimum {min_val}')
        if max_val is not None and v > max_val:
            raise ValueError(f'Value {v} exceeds maximum {max_val}')
            
        return v

class ConfigUpdate(BaseModel):
    """Configuration update request."""
    
    value: Union[int, str, float, bool] = Field(..., description="New configuration value")
    value_type: ValueType = Field(..., description="Type of the value")
    description: str = Field(..., description="Human-readable description")
    min_value: Optional[float] = Field(None, description="Minimum value for numeric types")
    max_value: Optional[float] = Field(None, description="Maximum value for numeric types")
```

## Data Models

### MongoDB Collections

#### SKUs Collection (`skus`)

**Document Structure:**

```json
{
  "_id": "50624",
  "tradeNumber": "50624",
  "customerName": "CHICK FIL A INC",
  "customerType": "FDS",
  "productType": "NUGGET",
  "unitsPerCut": 1,
  "prodPlant": "FSP",
  "minWeight": 10.0,
  "maxWeight": 19.0,
  "targetWeight": 15.0,
  "birdSize": "SB",
  "allowedParts": ["D"]
}
```

**Indexes:**
- Primary: `_id` (trade number)
- Secondary: `customerType`, `productType`, `prodPlant` (for search optimization)

#### Global Config Collection (`global_config`)

**Document Structure:**

```json
{
  "_id": "enumeration.defaultMaxTrim",
  "key": "enumeration.defaultMaxTrim",
  "value": 2,
  "valueType": "int",
  "description": "Default max trim allowed",
  "updatedAt": "2024-03-08T10:30:00Z",
  "minValue": 0,
  "maxValue": 100
}
```

**Indexes:**
- Primary: `_id` (configuration key)

### Field Mappings

**MongoDB to Python:**
- MongoDB uses camelCase field names (e.g., `tradeNumber`)
- Python models use snake_case (e.g., `trade_number`)
- Pydantic `alias` configuration handles the mapping

**Example:**

```python
class SKU(BaseModel):
    trade_number: str = Field(..., alias="tradeNumber")
    
    class Config:
        populate_by_name = True  # Allow both snake_case and camelCase
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Enumeration API Properties

**Property 1: SKU retrieval returns correct data**
*For any* SKU document stored in the database, retrieving it by trade number should return a document where the _id field equals the tradeNumber field, and the document contains all the expected data.
**Validates: Requirements 5.2, 7.1**

**Property 2: Search results match criteria**
*For any* search criteria provided, all returned SKU documents should match the specified filters (e.g., if searching by customerType="FDS", all results should have customerType="FDS").
**Validates: Requirements 6.2, 6.3**

**Property 3: SKU schema validation**
*For any* SKU document, it should contain all required fields (tradeNumber, customerName, customerType, productType, unitsPerCut, prodPlant, minWeight, maxWeight, targetWeight, birdSize, allowedParts), with minWeight/maxWeight/targetWeight as numeric values, and allowedParts as an array of strings.
**Validates: Requirements 7.2, 7.3, 7.4**

**Property 4: Weight range consistency**
*For any* SKU document, the minWeight should be less than maxWeight, and targetWeight should be between minWeight and maxWeight.
**Validates: Requirements 7.3**

### Global Config API Properties

**Property 5: Config retrieval returns correct data**
*For any* configuration document stored in the database, retrieving it by key should return a document where the _id field equals the key field.
**Validates: Requirements 10.2**

**Property 6: Config update persistence**
*For any* configuration update (create or modify), after the update operation completes, retrieving the configuration by key should return the updated value, and the updatedAt field should be set to a valid ISO 8601 timestamp that is greater than or equal to the update request time.
**Validates: Requirements 11.2, 11.3, 11.4**

**Property 7: Value type validation**
*For any* configuration document, the value field should match the type specified by valueType (int values for "int", string values for "string", float values for "float", bool values for "bool").
**Validates: Requirements 11.5**

**Property 8: Numeric range validation**
*For any* configuration document with numeric value and minValue/maxValue constraints, the value should be greater than or equal to minValue (if present) and less than or equal to maxValue (if present).
**Validates: Requirements 11.6**

**Property 9: Get all configs completeness**
*For any* state of the database, calling GET /config should return a list containing all configuration documents present in the global_config collection.
**Validates: Requirements 12.2**

**Property 10: Config schema validation**
*For any* configuration document, it should contain all required fields (key, value, valueType, description, updatedAt), with _id equal to key, valueType being one of the allowed values ("int", "string", "float", "bool"), and updatedAt being a valid ISO 8601 formatted string.
**Validates: Requirements 13.1, 13.2, 13.3, 13.5**

### Batch Operations Properties

**Property 11: Batch import validation**
*For any* batch import request, if any SKU in the batch fails validation, the entire batch should be rejected without inserting any SKUs, and the response should include error details for all failed SKUs.
**Validates: Requirements 18.2, 18.3**

**Property 12: Batch import success**
*For any* batch import request where all SKUs are valid, all SKUs should be successfully inserted into the database, and the response should indicate the correct count of successful imports.
**Validates: Requirements 18.1, 18.4**

**Property 13: Batch export completeness**
*For any* export request, the returned array should contain all SKU documents present in the database (or all matching the filter criteria if provided).
**Validates: Requirements 18.5, 18.6**

**Property 14: Batch config update validation**
*For any* batch configuration update request, if any config in the batch fails validation, the entire batch should be rejected without updating any configs, and the response should include error details for all failed configs.
**Validates: Requirements 18.8, 18.9**

**Property 15: Batch config update success**
*For any* batch configuration update request where all configs are valid, all configs should be successfully updated in the database, and the response should indicate the correct count of successful updates.
**Validates: Requirements 18.7, 18.10**

## Error Handling

### Enumeration API Error Handling

**HTTP Status Codes:**
- `200 OK`: Successful retrieval or search
- `404 Not Found`: SKU with specified trade number does not exist
- `422 Unprocessable Entity`: Invalid request body or validation error
- `500 Internal Server Error`: Database connection error or unexpected error

**Error Response Format:**

```python
class ErrorResponse(BaseModel):
    """Standard error response format."""
    
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Machine-readable error code")
```

**Error Scenarios:**

1. **SKU Not Found**: When GET /skus/{trade_number} is called with non-existent trade number
   - Return 404 with message: "SKU with trade number {trade_number} not found"

2. **Invalid Search Criteria**: When POST /skus/search receives invalid filter format
   - Return 422 with validation error details

3. **Database Connection Error**: When MongoDB is unavailable
   - Return 500 with message: "Database connection error"
   - Log full error details for debugging

4. **Schema Validation Error**: When SKU data doesn't match expected schema
   - Return 422 with Pydantic validation error details

### Global Config API Error Handling

**HTTP Status Codes:**
- `200 OK`: Successful retrieval, update, or creation
- `404 Not Found`: Configuration with specified key does not exist (GET only)
- `422 Unprocessable Entity`: Invalid request body or validation error
- `500 Internal Server Error`: Database connection error or unexpected error

**Error Scenarios:**

1. **Config Not Found**: When GET /config/{key} is called with non-existent key
   - Return 404 with message: "Configuration with key {key} not found"

2. **Value Type Mismatch**: When PUT /config/{key} receives value that doesn't match valueType
   - Return 422 with message: "Value must be of type {valueType}"

3. **Range Validation Error**: When PUT /config/{key} receives numeric value outside min/max range
   - Return 422 with message: "Value {value} must be between {minValue} and {maxValue}"

4. **Invalid ValueType**: When PUT /config/{key} receives unsupported valueType
   - Return 422 with message: "valueType must be one of: int, string, float, bool"

5. **Database Connection Error**: When MongoDB is unavailable
   - Return 500 with message: "Database connection error"
   - Log full error details for debugging

### Error Handling Implementation Pattern

Both APIs follow this error handling pattern:

```python
@router.get("/skus/{trade_number}")
async def get_sku(
    trade_number: str,
    service: SKUService = Depends(get_sku_service)
):
    """Get SKU by trade number.
    
    Raises:
        HTTPException: 404 if SKU not found, 500 for database errors
    """
    try:
        sku = service.get_sku_by_trade_number(trade_number)
        if sku is None:
            raise HTTPException(
                status_code=404,
                detail=f"SKU with trade number {trade_number} not found"
            )
        return sku
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error retrieving SKU: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
```

## Testing Strategy

### Overview

The testing strategy employs both unit testing and property-based testing to ensure comprehensive coverage:

- **Unit tests** verify specific examples, edge cases, and error conditions
- **Property tests** verify universal properties that should hold across all inputs
- Together they provide comprehensive coverage: unit tests catch concrete bugs, property tests verify general correctness

### Unit Testing

**Framework**: pytest

**Test Organization:**
- Tests are co-located with source files using `_test.py` suffix
- Example: `services/sku_service.py` → `services/sku_service_test.py`

**Unit Test Coverage:**

1. **Router Layer Tests**:
   - Test endpoint existence and HTTP methods
   - Test request/response serialization
   - Test error status codes (404, 422, 500)
   - Example: Test that GET /health returns 200 with {"status": "healthy"}

2. **Service Layer Tests**:
   - Test business logic with mock repositories
   - Test error handling and edge cases
   - Example: Test that searching with empty criteria returns all SKUs

3. **Repository Layer Tests**:
   - Test database queries with test database
   - Test document mapping to models
   - Example: Test that find_by_trade_number returns None for non-existent ID

4. **Model Validation Tests**:
   - Test Pydantic validation rules
   - Test field constraints and validators
   - Example: Test that SKU with maxWeight < minWeight raises ValidationError

**Example Unit Test:**

```python
def test_get_sku_not_found():
    """Test that getting non-existent SKU returns 404."""
    # Arrange
    mock_repo = Mock(spec=SKURepository)
    mock_repo.find_by_trade_number.return_value = None
    service = SKUService(mock_repo)
    
    # Act
    result = service.get_sku_by_trade_number("NONEXISTENT")
    
    # Assert
    assert result is None
```

### Property-Based Testing

**Framework**: Hypothesis (Python property-based testing library)

**Configuration**: Each property-based test runs a minimum of 100 iterations to ensure thorough coverage of the input space.

**Property Test Coverage:**

Each correctness property from the design document is implemented as a property-based test:

1. **Property 1: SKU retrieval returns correct data**
   - Generate random SKU documents
   - Insert into test database
   - Verify retrieval returns correct data with _id = tradeNumber
   - **Feature: fastapi-enumeration-services, Property 1: SKU retrieval returns correct data**

2. **Property 2: Search results match criteria**
   - Generate random SKU documents and search criteria
   - Insert SKUs into test database
   - Verify all search results match the criteria
   - **Feature: fastapi-enumeration-services, Property 2: Search results match criteria**

3. **Property 3: SKU schema validation**
   - Generate random SKU data
   - Verify Pydantic validation accepts valid SKUs
   - Verify Pydantic validation rejects invalid SKUs
   - **Feature: fastapi-enumeration-services, Property 3: SKU schema validation**

4. **Property 4: Weight range consistency**
   - Generate random weight values
   - Verify SKU validation enforces minWeight < maxWeight
   - **Feature: fastapi-enumeration-services, Property 4: Weight range consistency**

5. **Property 5: Config retrieval returns correct data**
   - Generate random config documents
   - Insert into test database
   - Verify retrieval returns correct data with _id = key
   - **Feature: fastapi-enumeration-services, Property 5: Config retrieval returns correct data**

6. **Property 6: Config update persistence**
   - Generate random config updates
   - Perform update operation
   - Verify retrieval returns updated value with valid updatedAt
   - **Feature: fastapi-enumeration-services, Property 6: Config update persistence**

7. **Property 7: Value type validation**
   - Generate random values and valueTypes
   - Verify validation accepts matching types
   - Verify validation rejects mismatched types
   - **Feature: fastapi-enumeration-services, Property 7: Value type validation**

8. **Property 8: Numeric range validation**
   - Generate random numeric values and min/max constraints
   - Verify validation accepts values within range
   - Verify validation rejects values outside range
   - **Feature: fastapi-enumeration-services, Property 8: Numeric range validation**

9. **Property 9: Get all configs completeness**
   - Generate random set of config documents
   - Insert into test database
   - Verify GET /config returns all documents
   - **Feature: fastapi-enumeration-services, Property 9: Get all configs completeness**

10. **Property 10: Config schema validation**
    - Generate random config data
    - Verify Pydantic validation accepts valid configs
    - Verify all required fields are present and correctly typed
    - **Feature: fastapi-enumeration-services, Property 10: Config schema validation**

11. **Property 11: Batch import validation**
    - Generate random batch of SKUs with some invalid
    - Verify batch is rejected if any SKU is invalid
    - Verify no SKUs are inserted when batch fails
    - **Feature: fastapi-enumeration-services, Property 11: Batch import validation**

12. **Property 12: Batch import success**
    - Generate random batch of valid SKUs
    - Verify all SKUs are inserted successfully
    - Verify response indicates correct success count
    - **Feature: fastapi-enumeration-services, Property 12: Batch import success**

13. **Property 13: Batch export completeness**
    - Generate random set of SKU documents
    - Insert into test database
    - Verify export returns all documents
    - **Feature: fastapi-enumeration-services, Property 13: Batch export completeness**

14. **Property 14: Batch config update validation**
    - Generate random batch of config updates with some invalid
    - Verify batch is rejected if any config is invalid
    - Verify no configs are updated when batch fails
    - **Feature: fastapi-enumeration-services, Property 14: Batch config update validation**

15. **Property 15: Batch config update success**
    - Generate random batch of valid config updates
    - Verify all configs are updated successfully
    - Verify response indicates correct success count
    - **Feature: fastapi-enumeration-services, Property 15: Batch config update success**

**Example Property Test:**

```python
from hypothesis import given, strategies as st

@given(
    trade_number=st.text(min_size=1, max_size=20),
    customer_name=st.text(min_size=1),
    min_weight=st.floats(min_value=0, max_value=1000),
    max_weight=st.floats(min_value=0, max_value=1000)
)
def test_property_sku_retrieval(trade_number, customer_name, min_weight, max_weight):
    """Feature: fastapi-enumeration-services, Property 1: SKU retrieval returns correct data
    
    For any SKU document stored in the database, retrieving it by trade number
    should return a document where the _id field equals the tradeNumber field.
    """
    # Ensure min_weight < max_weight
    if min_weight >= max_weight:
        min_weight, max_weight = max_weight, min_weight + 1
    
    # Arrange: Create SKU document
    sku_data = {
        "tradeNumber": trade_number,
        "customerName": customer_name,
        "minWeight": min_weight,
        "maxWeight": max_weight,
        # ... other required fields
    }
    
    # Act: Insert and retrieve
    repository.insert(sku_data)
    retrieved = repository.find_by_trade_number(trade_number)
    
    # Assert: _id equals tradeNumber
    assert retrieved["_id"] == retrieved["tradeNumber"]
    assert retrieved["tradeNumber"] == trade_number
```

### Integration Testing

**Scope**: End-to-end API tests using TestClient

**Coverage**:
- Test complete request/response cycles
- Test API with real MongoDB test database
- Test error scenarios and edge cases

**Example Integration Test:**

```python
from fastapi.testclient import TestClient

def test_sku_crud_flow():
    """Test complete SKU retrieval flow."""
    client = TestClient(app)
    
    # Arrange: Insert test SKU into database
    test_sku = {...}
    test_db.skus.insert_one(test_sku)
    
    # Act: Call API endpoint
    response = client.get(f"/skus/{test_sku['tradeNumber']}")
    
    # Assert: Verify response
    assert response.status_code == 200
    assert response.json()["tradeNumber"] == test_sku["tradeNumber"]
```

### Test Environment

**Test Database**:
- Use separate MongoDB database for testing
- Database name: `enumeration_test_db` and `config_test_db`
- Clean database before each test run

**Test Configuration**:
- Use environment variables to configure test database connection
- Example `.env.test` file:
  ```
  MONGODB_URL=mongodb://root:password@localhost:27017
  MONGODB_DATABASE=enumeration_test_db
  ```

**Running Tests**:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run only property tests
pytest -m property

# Run only unit tests
pytest -m unit
```

## Deployment

### Docker Configuration

**Directory Structure:**

Both microservices follow a consistent package structure to ensure an easier development experience:

```
project-root/
├── docker-compose.yml
├── README.md
├── enumeration-api/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── .env.example
│   ├── README.md                    # Service-specific documentation
│   ├── openapi.json                 # OpenAPI specification
│   ├── main.py                      # FastAPI application entry point
│   ├── config.py                    # Configuration management
│   ├── database.py                  # Database connection
│   ├── models/
│   │   ├── __init__.py
│   │   └── sku.py                   # Pydantic models
│   ├── repositories/
│   │   ├── __init__.py
│   │   └── sku_repository.py        # Data access layer
│   ├── services/
│   │   ├── __init__.py
│   │   └── sku_service.py           # Business logic layer
│   └── routers/
│       ├── __init__.py
│       └── sku_router.py            # API endpoints
├── global-config-api/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── .env.example
│   ├── README.md                    # Service-specific documentation
│   ├── openapi.json                 # OpenAPI specification
│   ├── main.py                      # FastAPI application entry point
│   ├── config.py                    # Configuration management
│   ├── database.py                  # Database connection
│   ├── models/
│   │   ├── __init__.py
│   │   └── config.py                # Pydantic models
│   ├── repositories/
│   │   ├── __init__.py
│   │   └── config_repository.py     # Data access layer
│   ├── services/
│   │   ├── __init__.py
│   │   └── config_service.py        # Business logic layer
│   └── routers/
│       ├── __init__.py
│       └── config_router.py         # API endpoints
```

**Package Structure Consistency:**

Each microservice follows the same organizational pattern:
- `main.py`: Application entry point with FastAPI app initialization
- `config.py`: Settings and configuration using Pydantic BaseSettings
- `database.py`: MongoDB connection management and dependency injection
- `models/`: Pydantic models for request/response validation
- `repositories/`: Data access layer with MongoDB operations
- `services/`: Business logic layer
- `routers/`: API endpoint definitions
- `README.md`: Service-specific documentation with API usage examples
- `openapi.json`: OpenAPI 3.0 specification for the service API

### Dockerfile Pattern

Both services use the same Dockerfile pattern:

```dockerfile
# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (8000 for enumeration, 8001 for config)
EXPOSE 8000

# Run uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Python Dependencies

Both services use the same `requirements.txt`:

```
# Core FastAPI dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0

# Database
pymongo==4.6.0

# Environment variables
python-dotenv==1.0.0

# Observability
prometheus-client==0.19.0
opentelemetry-api==1.21.0
opentelemetry-sdk==1.21.0
opentelemetry-instrumentation-fastapi==0.42b0
opentelemetry-instrumentation-pymongo==0.42b0
opentelemetry-exporter-jaeger==1.21.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
hypothesis==6.92.1
```

### Docker Compose Configuration

**Services:**
1. MongoDB with authentication and persistent volume
2. Enumeration API on port 8000
3. Global Config API on port 8001

**Network**: All services on `mongo_net` bridge network

**Volumes**: `mongo_data` for MongoDB persistence

**Environment Variables**:
- MongoDB credentials (MONGO_INITDB_ROOT_USERNAME, MONGO_INITDB_ROOT_PASSWORD)
- Connection strings for each API service

### Environment Variables

**Enumeration API (.env):**
```
MONGODB_URL=mongodb://root:password@mongodb:27017
MONGODB_DATABASE=enumeration_db
JAEGER_HOST=jaeger
JAEGER_PORT=6831
```

**Global Config API (.env):**
```
MONGODB_URL=mongodb://root:password@mongodb:27017
MONGODB_DATABASE=config_db
JAEGER_HOST=jaeger
JAEGER_PORT=6831
```

### Startup and Health Checks

**Service Startup Order:**
1. MongoDB starts first
2. API services wait for MongoDB to be ready (using depends_on with health check)
3. API services connect to MongoDB on startup

**Health Check Endpoints:**
- Enumeration API: `http://localhost:8000/health`
- Global Config API: `http://localhost:8001/health`
- MongoDB: Internal health check in docker-compose

### Running the System

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

## API Documentation

### Service README Files

Each microservice includes a comprehensive README.md file that documents:

1. **Service Overview**: Purpose and capabilities of the service
2. **Architecture**: Layered architecture explanation (router → service → repository)
3. **API Endpoints**: Complete list of endpoints with descriptions
4. **Request/Response Examples**: Sample requests and responses for each endpoint
5. **Setup Instructions**: How to run the service locally and with Docker
6. **Environment Variables**: Required configuration
7. **Testing**: How to run tests
8. **Dependencies**: List of Python packages and their purposes

### OpenAPI Specification

Each microservice provides an OpenAPI 3.0 specification file (`openapi.json`) that includes:

- Complete API endpoint definitions
- Request/response schemas
- Parameter descriptions
- Error response formats
- Example values
- Authentication requirements (if applicable)

The OpenAPI spec can be:
- Used to generate client SDKs
- Imported into API testing tools (Postman, Insomnia)
- Used for API documentation generation
- Validated against the implementation

### Automatic Interactive Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs` (Enumeration API)
- **Swagger UI**: `http://localhost:8001/docs` (Global Config API)
- **ReDoc**: `http://localhost:8000/redoc` (Enumeration API)
- **ReDoc**: `http://localhost:8001/redoc` (Global Config API)

These interactive docs allow:
- Browsing all endpoints
- Viewing request/response schemas
- Testing endpoints directly from the browser
- Downloading the OpenAPI specification

### Generating OpenAPI Specification

The `openapi.json` file can be generated from the running application:

```python
# In main.py or a separate script
import json
from main import app

# Generate OpenAPI schema
openapi_schema = app.openapi()

# Save to file
with open("openapi.json", "w") as f:
    json.dump(openapi_schema, f, indent=2)
```

Or via command line:
```bash
# Start the service and download the spec
curl http://localhost:8000/openapi.json > enumeration-api/openapi.json
curl http://localhost:8001/openapi.json > global-config-api/openapi.json
```

### Example API Usage

**Enumeration API Examples:**

```bash
# Health check
curl http://localhost:8000/health

# Get SKU by trade number
curl http://localhost:8000/skus/50624

# Search SKUs
curl -X POST http://localhost:8000/skus/search \
  -H "Content-Type: application/json" \
  -d '{"customerType": "FDS", "productType": "NUGGET"}'
```

**Global Config API Examples:**

```bash
# Health check
curl http://localhost:8001/health

# Get all configs
curl http://localhost:8001/config

# Get specific config
curl http://localhost:8001/config/enumeration.defaultMaxTrim

# Update config
curl -X PUT http://localhost:8001/config/enumeration.defaultMaxTrim \
  -H "Content-Type: application/json" \
  -d '{
    "value": 5,
    "valueType": "int",
    "description": "Default max trim allowed",
    "minValue": 0,
    "maxValue": 100
  }'
```

## Security Considerations

### MongoDB Authentication

- MongoDB requires username/password authentication
- Credentials are configured via environment variables
- Connection strings include credentials: `mongodb://user:pass@host:port`

### API Security (Future Enhancements)

The current design focuses on internal microservices communication. For production deployment, consider:

1. **API Authentication**: Add JWT or API key authentication
2. **HTTPS**: Use TLS certificates for encrypted communication
3. **Rate Limiting**: Implement rate limiting to prevent abuse
4. **Input Validation**: Pydantic models provide input validation
5. **CORS**: Configure CORS policies for web clients

### Environment Variables

- Never commit `.env` files with real credentials
- Use `.env.example` files as templates
- In production, use secrets management (e.g., Docker secrets, Kubernetes secrets)

## Performance Considerations

### Database Indexing

**SKUs Collection:**
- Primary index on `_id` (trade number) - automatic
- Secondary indexes on frequently queried fields:
  - `customerType`
  - `productType`
  - `prodPlant`

**Global Config Collection:**
- Primary index on `_id` (key) - automatic
- No secondary indexes needed (small collection)

### Connection Pooling

PyMongo automatically manages connection pooling:
- Default pool size: 100 connections
- Connections are reused across requests
- Configure via `maxPoolSize` in connection string if needed

### Caching Strategy (Future Enhancement)

For high-traffic scenarios, consider:
- Redis cache for frequently accessed SKUs
- Cache invalidation on updates
- TTL-based cache expiration

### Monitoring

Recommended monitoring metrics:
- API response times (p50, p95, p99)
- Database query times
- Error rates by endpoint
- Active database connections
- Container resource usage (CPU, memory)

## Maintenance and Operations

### Logging

**Log Levels:**
- INFO: Normal operations (startup, shutdown, requests)
- WARNING: Recoverable errors (validation failures, not found)
- ERROR: Unexpected errors (database connection failures)

**Log Format:**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Database Backups

MongoDB data persists in Docker volume `mongo_data`:
- Regular backups using `mongodump`
- Restore using `mongorestore`
- Consider automated backup schedules in production

### Updating Services

```bash
# Rebuild and restart specific service
docker-compose up -d --build enumeration-api

# Rebuild all services
docker-compose up -d --build
```

### Database Migrations

For schema changes:
1. Create migration scripts using PyMongo
2. Run migrations before deploying new API version
3. Test migrations on staging environment first

## Future Enhancements

### Potential Improvements

1. **Authentication & Authorization**
   - Add JWT-based authentication
   - Implement role-based access control (RBAC)

2. **Advanced Search**
   - Full-text search on SKU fields
   - Fuzzy matching for customer names
   - Pagination for large result sets

3. **Caching Layer**
   - Redis cache for frequently accessed data
   - Cache warming strategies

4. **API Versioning**
   - Version API endpoints (/v1/skus, /v2/skus)
   - Maintain backward compatibility

5. **Webhooks**
   - Notify external systems of configuration changes
   - Event-driven architecture

6. **GraphQL API**
   - Alternative to REST for flexible queries
   - Reduce over-fetching of data

## Observability

### Metrics Collection

**Framework**: Prometheus client library for Python

**Metrics Exposed:**

1. **Request Metrics**:
   - `http_requests_total`: Counter of total HTTP requests by method, endpoint, and status
   - `http_request_duration_seconds`: Histogram of request duration by endpoint
   - `http_requests_in_progress`: Gauge of currently processing requests

2. **Database Metrics**:
   - `mongodb_connections_active`: Gauge of active MongoDB connections
   - `mongodb_query_duration_seconds`: Histogram of query execution time
   - `mongodb_errors_total`: Counter of database errors

3. **Business Metrics**:
   - `skus_total`: Gauge of total SKUs in database
   - `configs_total`: Gauge of total configurations
   - `batch_operations_total`: Counter of batch operations by type and status

**Metrics Endpoint:**
- Enumeration API: `GET /metrics`
- Global Config API: `GET /metrics`

**Implementation Pattern:**

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response

# Define metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['endpoint']
)

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )

# Middleware to track metrics
@app.middleware("http")
async def track_metrics(request: Request, call_next):
    """Track request metrics."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    http_requests_total.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    http_request_duration.labels(
        endpoint=request.url.path
    ).observe(duration)
    
    return response
```

### Structured Logging

**Framework**: Python `logging` module with JSON formatter

**Log Format:**

```json
{
  "timestamp": "2024-03-08T10:30:00.123Z",
  "level": "INFO",
  "service": "enumeration-api",
  "trace_id": "abc123",
  "span_id": "def456",
  "message": "SKU retrieved successfully",
  "method": "GET",
  "path": "/skus/50624",
  "status_code": 200,
  "duration_ms": 45
}
```

**Implementation:**

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """Format logs as JSON."""
    
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "service": "enumeration-api",  # or "global-config-api"
            "message": record.getMessage(),
        }
        
        # Add trace context if available
        if hasattr(record, 'trace_id'):
            log_data['trace_id'] = record.trace_id
        if hasattr(record, 'span_id'):
            log_data['span_id'] = record.span_id
            
        # Add request context if available
        if hasattr(record, 'method'):
            log_data['method'] = record.method
        if hasattr(record, 'path'):
            log_data['path'] = record.path
        if hasattr(record, 'status_code'):
            log_data['status_code'] = record.status_code
        if hasattr(record, 'duration_ms'):
            log_data['duration_ms'] = record.duration_ms
            
        return json.dumps(log_data)

# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

### Distributed Tracing

**Framework**: OpenTelemetry

**Components:**
- OpenTelemetry SDK for Python
- FastAPI instrumentation
- PyMongo instrumentation
- Trace exporter (Jaeger, Zipkin, or OTLP)

**Trace Context:**
- Trace ID: Unique identifier for the entire request flow
- Span ID: Unique identifier for each operation within a trace
- Parent Span ID: Links spans in a hierarchy

**Implementation:**

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.pymongo import PymongoInstrumentor

# Initialize tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Configure exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)
span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

# Instrument PyMongo
PymongoInstrumentor().instrument()

# Manual span creation for custom operations
@router.get("/skus/{trade_number}")
async def get_sku(trade_number: str):
    with tracer.start_as_current_span("get_sku") as span:
        span.set_attribute("trade_number", trade_number)
        # ... operation logic
        span.set_attribute("result", "success")
```

**Docker Compose Integration:**

Add Jaeger service to docker-compose.yml:

```yaml
services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # Jaeger UI
      - "6831:6831/udp"  # Jaeger agent
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411
```

### Monitoring Dashboard

**Recommended Stack:**
- Prometheus: Metrics collection and storage
- Grafana: Visualization and dashboards
- Jaeger: Distributed tracing UI

**Key Dashboards:**
1. **Service Health**: Request rate, error rate, latency percentiles
2. **Database Performance**: Connection pool usage, query duration
3. **Business Metrics**: SKU count trends, config update frequency
4. **Trace Analysis**: Request flow visualization, bottleneck identification

## Batch Operations

### Batch Import SKUs

**Endpoint**: `POST /skus/batch`

**Request Model:**

```python
class BatchImportRequest(BaseModel):
    """Batch import request."""
    
    skus: List[SKU] = Field(..., description="List of SKUs to import")
    validate_only: bool = Field(False, description="Only validate, don't insert")

class BatchImportResult(BaseModel):
    """Batch import result."""
    
    total: int = Field(..., description="Total SKUs in request")
    successful: int = Field(..., description="Successfully imported")
    failed: int = Field(..., description="Failed to import")
    errors: List[Dict[str, str]] = Field(..., description="Error details")
```

**Service Implementation:**

```python
class SKUService:
    def batch_import(self, skus: List[SKU], validate_only: bool = False) -> BatchImportResult:
        """Import multiple SKUs.
        
        Args:
            skus: List of SKU models to import
            validate_only: If True, only validate without inserting
            
        Returns:
            BatchImportResult with success/failure counts
            
        Behavior:
            1. Validate all SKUs first
            2. If any validation fails, return errors without inserting
            3. If all valid and not validate_only, insert all SKUs
            4. Return summary of results
        """
        errors = []
        
        # Validate all SKUs
        for idx, sku in enumerate(skus):
            try:
                # Pydantic validation happens automatically
                # Additional business logic validation
                if self.repository.find_by_trade_number(sku.trade_number):
                    errors.append({
                        "index": idx,
                        "trade_number": sku.trade_number,
                        "error": "SKU already exists"
                    })
            except ValidationError as e:
                errors.append({
                    "index": idx,
                    "trade_number": sku.trade_number,
                    "error": str(e)
                })
        
        # If validation errors or validate_only, return without inserting
        if errors or validate_only:
            return BatchImportResult(
                total=len(skus),
                successful=0,
                failed=len(errors),
                errors=errors
            )
        
        # Insert all SKUs
        successful = 0
        for sku in skus:
            try:
                self.repository.insert(sku.dict(by_alias=True))
                successful += 1
            except Exception as e:
                errors.append({
                    "trade_number": sku.trade_number,
                    "error": str(e)
                })
        
        return BatchImportResult(
            total=len(skus),
            successful=successful,
            failed=len(errors),
            errors=errors
        )
```

### Batch Export SKUs

**Endpoint**: `GET /skus/export`

**Query Parameters:**
- `format`: Export format (json, csv) - default: json
- `filter`: Optional filter criteria (JSON string)

**Response:**
- JSON: Array of SKU documents
- CSV: Comma-separated values with headers

**Service Implementation:**

```python
class SKUService:
    def export_all(self, filter_criteria: Optional[Dict] = None) -> List[SKU]:
        """Export all SKUs matching criteria.
        
        Args:
            filter_criteria: Optional MongoDB filter
            
        Returns:
            List of all matching SKU models
            
        Note:
            For large datasets, consider streaming or pagination
        """
        documents = self.repository.find_by_criteria(filter_criteria or {})
        return [SKU(**doc) for doc in documents]
```

### Batch Update Configurations

**Endpoint**: `POST /config/batch`

**Request Model:**

```python
class BatchConfigUpdate(BaseModel):
    """Batch configuration update."""
    
    key: str
    update: ConfigUpdate

class BatchUpdateRequest(BaseModel):
    """Batch update request."""
    
    configs: List[BatchConfigUpdate] = Field(..., description="Configurations to update")
    validate_only: bool = Field(False, description="Only validate, don't update")

class BatchUpdateResult(BaseModel):
    """Batch update result."""
    
    total: int = Field(..., description="Total configs in request")
    successful: int = Field(..., description="Successfully updated")
    failed: int = Field(..., description="Failed to update")
    errors: List[Dict[str, str]] = Field(..., description="Error details")
```

**Service Implementation:**

```python
class ConfigService:
    def batch_update(
        self,
        updates: List[BatchConfigUpdate],
        validate_only: bool = False
    ) -> BatchUpdateResult:
        """Update multiple configurations.
        
        Args:
            updates: List of configuration updates
            validate_only: If True, only validate without updating
            
        Returns:
            BatchUpdateResult with success/failure counts
            
        Behavior:
            1. Validate all updates first
            2. If any validation fails, return errors without updating
            3. If all valid and not validate_only, update all configs
            4. Return summary of results
        """
        errors = []
        
        # Validate all updates
        for idx, batch_update in enumerate(updates):
            try:
                # Validate the update model
                update = batch_update.update
                # Additional validation logic
                self._validate_config_update(batch_update.key, update)
            except ValidationError as e:
                errors.append({
                    "index": idx,
                    "key": batch_update.key,
                    "error": str(e)
                })
        
        # If validation errors or validate_only, return without updating
        if errors or validate_only:
            return BatchUpdateResult(
                total=len(updates),
                successful=0,
                failed=len(errors),
                errors=errors
            )
        
        # Update all configs
        successful = 0
        for batch_update in updates:
            try:
                self.update_config(batch_update.key, batch_update.update)
                successful += 1
            except Exception as e:
                errors.append({
                    "key": batch_update.key,
                    "error": str(e)
                })
        
        return BatchUpdateResult(
            total=len(updates),
            successful=successful,
            failed=len(errors),
            errors=errors
        )
```

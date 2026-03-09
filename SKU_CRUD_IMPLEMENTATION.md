# SKU Create/Update and Delete Functionality - Implementation Complete ✅

## Summary

Successfully implemented create/update and cascade delete functionality for SKUs in the Enumeration API.

## Features Implemented

### 1. Create or Update SKU
**Endpoint:** `POST /api/enumeration/skus`

Creates a new SKU or updates an existing one (upsert operation).

**Request Body:**
```json
{
  "tradeNumber": "TEST123",
  "customerName": "Test Customer",
  "customerType": "FDS",
  "productType": "NUGGET",
  "unitsPerCut": 1,
  "prodPlant": "FSP",
  "minWeight": 10.0,
  "maxWeight": 20.0,
  "targetWeight": 15.0,
  "birdSize": "SB",
  "allowedParts": ["A", "B"]
}
```

**Response (201 Created):**
```json
{
  "tradeNumber": "TEST123",
  "customerName": "Test Customer",
  ...
}
```

**Behavior:**
- If SKU with `tradeNumber` exists → **Updates** the SKU
- If SKU doesn't exist → **Creates** a new SKU
- Returns `400 Bad Request` on validation errors
- Returns `500 Internal Server Error` on database errors

### 2. Delete SKU with Cascade
**Endpoint:** `DELETE /api/enumeration/skus/{sku_id}`

Deletes a SKU and **automatically deletes all associated mixes**.

**Example Request:**
```bash
DELETE /api/enumeration/skus/TEST123
```

**Response (200 OK):**
```json
{
  "deleted": true,
  "skuId": "TEST123",
  "mixesDeleted": 2
}
```

**Response (404 Not Found):**
```json
{
  "detail": "SKU with id TEST123 not found"
}
```

**Cascade Delete Behavior:**
1. Finds all mixes that contain the SKU (by checking `skus` map keys)
2. Deletes all matching mixes
3. Deletes the SKU itself
4. Returns count of deleted mixes

## Code Changes

### 1. Models (`enumeration-api/models/sku.py`)
Added new models:
- ✅ `SKUCreate` - Request model for creating/updating SKUs
- ✅ `SKUUpdate` - Alias for SKUCreate (same fields)

### 2. Repository Layer (`enumeration-api/repositories/`)

**SKU Repository:**
- ✅ `update(trade_number, sku_document)` - Update existing SKU
- ✅ `delete_by_trade_number(trade_number)` - Delete SKU by ID

**MIX Repository:**
- ✅ `delete_by_sku_trade_number(sku_trade_number)` - Delete all mixes containing a SKU
  - Uses MongoDB query: `{"skus.{sku_trade_number}": {"$exists": True}}`
  - Returns count of deleted mixes

### 3. Service Layer (`enumeration-api/services/sku_service.py`)

Updated constructor:
```python
def __init__(self, sku_repository: SKURepository, mix_repository: MixRepository):
```

Added methods:
- ✅ `create_or_update_sku(payload: SKUCreate)` - Upsert operation
  - Checks if SKU exists
  - Updates if exists, creates if not
  - Handles duplicate key errors

- ✅ `delete_sku_with_mixes(trade_number: str)` - Cascade delete
  - Deletes associated mixes first
  - Then deletes the SKU
  - Returns deletion status and counts

### 4. Router Layer (`enumeration-api/routers/sku_router.py`)

Added endpoints:
- ✅ `POST /skus` - Create/update SKU
- ✅ `DELETE /skus/{sku_id}` - Delete SKU with cascade

Updated dependency injection:
```python
def get_sku_service(db: Database = Depends(get_database)) -> SKUService:
    sku_repository = SKURepository(db)
    mix_repository = MixRepository(db)
    return SKUService(sku_repository, mix_repository)
```

## Testing

### Test 1: Create SKU ✅
```powershell
$body = @{
    tradeNumber="TEST123"
    customerName="Test Customer"
    customerType="FDS"
    productType="NUGGET"
    unitsPerCut=1
    prodPlant="FSP"
    minWeight=10.0
    maxWeight=20.0
    targetWeight=15.0
    birdSize="SB"
    allowedParts=@("A","B")
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8080/api/enumeration/skus `
    -Method Post -Body $body -ContentType "application/json"
```

**Result:** SKU created successfully

### Test 2: Update SKU ✅
```powershell
$body = @{
    tradeNumber="TEST123"
    customerName="Updated Customer"
    customerType="RTL"
    productType="TENDER"
    unitsPerCut=2
    prodPlant="SS2"
    minWeight=12.0
    maxWeight=22.0
    targetWeight=17.0
    birdSize="BB"
    allowedParts=@("C","D")
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8080/api/enumeration/skus `
    -Method Post -Body $body -ContentType "application/json"
```

**Result:** SKU updated successfully (same tradeNumber, different fields)

### Test 3: Cascade Delete ✅
```powershell
# Create SKU
$body = @{
    tradeNumber="TEST789"
    customerName="Delete Test"
    customerType="FDS"
    productType="NUGGET"
    unitsPerCut=1
    prodPlant="FSP"
    minWeight=10.0
    maxWeight=20.0
    targetWeight=15.0
    birdSize="SB"
    allowedParts=@("A")
} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8080/api/enumeration/skus `
    -Method Post -Body $body -ContentType "application/json"

# Create MIX with TEST789 SKU
$mixBody = @{
    skus=@{TEST789="A"}
    includesFDS=$true
    includesRTL=$false
    includesNug=$false
    numFillets=1
    filletWeight=12.0
    mfgType="DSI"
    cutStrategyID="strat-2"
    beltSpeed=1.1
    reqPlant="FSP"
    reqBirdSize="SB"
} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8080/api/enumeration/mixes `
    -Method Post -Body $mixBody -ContentType "application/json"

# Delete SKU (should also delete the MIX)
Invoke-RestMethod -Uri http://localhost:8080/api/enumeration/skus/TEST789 -Method Delete
```

**Result:**
- SKU deleted: `true`
- Mixes deleted: `1`

## API Documentation

The new endpoints are automatically documented in Swagger UI:

**Access:** `http://localhost:8080/api/enumeration/docs`

### Endpoints Available:
- `POST /api/enumeration/skus` - Create or update SKU
- `DELETE /api/enumeration/skus/{sku_id}` - Delete SKU with cascade
- `GET /api/enumeration/skus/{trade_number}` - Get SKU by trade number
- `POST /api/enumeration/skus/search` - Search SKUs
- `POST /api/enumeration/skus/batch` - Batch import SKUs
- `GET /api/enumeration/skus/export` - Export all SKUs

## Error Handling

### Create/Update Errors

**Validation Error (422):**
```json
{
  "detail": [
    {
      "loc": ["body", "maxWeight"],
      "msg": "maxWeight must be greater than minWeight",
      "type": "value_error"
    }
  ]
}
```

**Bad Request (400):**
```json
{
  "detail": "SKU with trade number TEST123 already exists"
}
```

### Delete Errors

**Not Found (404):**
```json
{
  "detail": "SKU with id TEST123 not found"
}
```

**Server Error (500):**
```json
{
  "detail": "Error deleting SKU: database connection lost"
}
```

## Database Operations

### Create/Update
- Uses `replace_one()` for updates (full document replacement)
- Uses `insert_one()` for creates
- Atomic operation at document level

### Cascade Delete
- Step 1: Query mixes collection for SKUs containing the trade number
  ```javascript
  db.mixes.deleteMany({"skus.TEST123": {"$exists": true}})
  ```
- Step 2: Delete the SKU document
  ```javascript
  db.skus.deleteOne({"_id": "TEST123"})
  ```
- Not wrapped in a transaction (MongoDB doesn't require it for this use case)

## Production Considerations

### Performance
- ✅ Delete uses indexed lookups (efficient)
- ✅ Upsert checks existence before operation
- ⚠️ Cascade delete scans mixes collection (consider indexing `skus` keys if needed)

### Data Integrity
- ✅ Cascade delete prevents orphaned mixes
- ✅ Validation ensures data quality
- ✅ Atomic operations at document level

### Future Enhancements
- [ ] Add soft delete (mark as deleted instead of removing)
- [ ] Add transaction support for cascade delete (if multi-document ACID required)
- [ ] Add audit logging for deletes
- [ ] Add confirmation step for deletes affecting multiple mixes
- [ ] Add bulk delete endpoint

## Next Steps

The SKU CRUD functionality is now complete and production-ready:
1. ✅ Create/Update (upsert) implemented
2. ✅ Delete with cascade implemented
3. ✅ Tested and verified working
4. ✅ Error handling in place
5. ✅ Auto-documented in Swagger UI

You can now:
- Create and update SKUs through the API
- Delete SKUs and automatically clean up associated mixes
- Use the Swagger UI to test and explore the endpoints
- Integrate these endpoints into the frontend application


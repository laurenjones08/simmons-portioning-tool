# MIX Configuration

## Overview

The MIX model represents a combination of SKUs used together for portioning decisions. It includes references to required production plants, bird sizes, and manufacturing line types, which are all configured centrally in the Global Config API.

## MIX Model Fields

### Core Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `_id` / `mix_id` | ObjectId (string) | Generated | Unique identifier |
| `skus` | Dict[str, str] | Yes | Map of SKU trade numbers to part IDs |
| `mfgType` | Enum: "DSI", "DB20" | Yes | Manufacturing line type |
| `reqPlant` | String | Yes | Required production plant |
| `reqBirdSize` | String | Yes | Required bird size |

### Mix Composition

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `includesFDS` | Boolean | Yes | Includes food service customer |
| `includesRTL` | Boolean | Yes | Includes retail customer |
| `includesNug` | Boolean | Yes | Includes nugget SKU |
| `nuggetTargetWeight` | Float (nullable) | Conditional | Target weight per nugget (>0 if includesNug=true, null otherwise) |
| `numFillets` | Integer | Yes | Count of fillet SKUs |
| `filletWeight` | Float | Yes | Total weight of fillets |

### Configuration Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `cutStrategyID` | String | Yes | Predetermined cut strategy |
| `beltSpeed` | Float | Yes | Required belt speed |

### Internal Fields

| Field | Type | Purpose |
|-------|------|---------|
| `skuSetKey` | String | Derived key for uniqueness enforcement |

## Configuration Management

### Available Plants

Retrieve available production plants:

```bash
curl http://localhost:8080/api/config/mix.availablePlants
```

Response:
```json
{
  "key": "mix.availablePlants",
  "value": "FSP,SS2,VBS",
  "valueType": "string",
  "description": "Comma-separated list of available production plants",
  "updatedAt": "2026-03-09T12:00:00Z"
}
```

**Current Options:** FSP, SS2, VBS

### Available Bird Sizes

Retrieve available bird size classifications:

```bash
curl http://localhost:8080/api/config/mix.availableBirdSizes
```

Response:
```json
{
  "key": "mix.availableBirdSizes",
  "value": "SB,BB",
  "valueType": "string",
  "description": "Comma-separated list of available bird sizes",
  "updatedAt": "2026-03-09T12:00:00Z"
}
```

**Current Options:** SB (Small Bird), BB (Big Bird)

### Available Manufacturing Types

Retrieve available manufacturing line types:

```bash
curl http://localhost:8080/api/config/mix.availableMfgTypes
```

Response:
```json
{
  "key": "mix.availableMfgTypes",
  "value": "DSI,DB20",
  "valueType": "string",
  "description": "Comma-separated list of available manufacturing line types",
  "updatedAt": "2026-03-09T12:00:00Z"
}
```

**Current Options:** DSI, DB20

## Creating a MIX

### Request Format

```http
POST /api/enumeration/mixes
Content-Type: application/json

{
  "skus": {
    "123": "A",
    "456": "B",
    "567": "C"
  },
  "includesFDS": true,
  "includesRTL": false,
  "includesNug": true,
  "nuggetTargetWeight": 15.5,
  "numFillets": 2,
  "filletWeight": 12.75,
  "mfgType": "DSI",
  "cutStrategyID": "strategy-001",
  "beltSpeed": 1.2,
  "reqPlant": "FSP",
  "reqBirdSize": "SB"
}
```

### Response (201 Created)

```json
{
  "_id": "65f0c8fd6fb6bd463e25d4b7",
  "skus": {"123": "A", "456": "B", "567": "C"},
  "includesFDS": true,
  "includesRTL": false,
  "includesNug": true,
  "nuggetTargetWeight": 15.5,
  "numFillets": 2,
  "filletWeight": 12.75,
  "mfgType": "DSI",
  "cutStrategyID": "strategy-001",
  "beltSpeed": 1.2,
  "reqPlant": "FSP",
  "reqBirdSize": "SB"
}
```

## Uniqueness Rules

### One Mix Per SKU Set + Manufacturing Type

**Rule:** Only one MIX can exist for each unique combination of:
1. **SKU trade-number set** - determined by the keys in the `skus` map
2. **Manufacturing type** - `mfgType` field

**Examples:**

✅ **Allowed:**
- Mix 1: SKUs {123,456,789} with DSI
- Mix 2: SKUs {123,456,789} with DB20
(Same SKU set, different manufacturing types)

❌ **Rejected:**
- Mix 1: SKUs {123,456,789} with DSI
- Mix 2: SKUs {123,456,789} with DSI
(Duplicate SKU set + manufacturing type)

**Part ID Variance is OK:**
- Mix 1: {123→A, 456→B, 789→C} with DSI
- Mix 2: {123→X, 456→Y, 789→Z} with DSI
(These would conflict - same SKU set + mfgType)

### Conflict Response

When attempting to create/update a duplicate:

```http
409 Conflict
Content-Type: application/json

{
  "detail": "A mix already exists for this SKU set and mfgType"
}
```

## Validation Rules

### SKU Map Validation
- All keys and values must be non-empty strings
- No duplicate keys
- At least one SKU required

### Nugget Weight Rules
- If `includesNug = true`:
  - `nuggetTargetWeight` MUST be > 0
  - Cannot be null or zero

- If `includesNug = false`:
  - `nuggetTargetWeight` MUST be null
  - Cannot have a value

### Field Constraints
| Field | Constraint |
|-------|-----------|
| `numFillets` | >= 0 |
| `filletWeight` | >= 0.0 |
| `beltSpeed` | >= 0.0 |
| `cutStrategyID` | 1-100 characters |
| `reqPlant` | 1-50 characters |
| `reqBirdSize` | 1-50 characters |

## Frontend Integration

### Populating Dropdown Options

```javascript
async function getConfigOptions() {
  const plants = await fetch('/api/config/mix.availablePlants')
    .then(r => r.json())
    .then(config => config.value.split(','));

  const birdSizes = await fetch('/api/config/mix.availableBirdSizes')
    .then(r => r.json())
    .then(config => config.value.split(','));

  const mfgTypes = await fetch('/api/config/mix.availableMfgTypes')
    .then(r => r.json())
    .then(config => config.value.split(','));

  return { plants, birdSizes, mfgTypes };
}

// Usage:
const { plants, birdSizes, mfgTypes } = await getConfigOptions();
// plants: ["FSP", "SS2", "VBS"]
// birdSizes: ["SB", "BB"]
// mfgTypes: ["DSI", "DB20"]
```

### Validating Before Submit

```javascript
function validateMixForm(formData) {
  const errors = [];

  // Validate SKU set
  if (!formData.skus || Object.keys(formData.skus).length === 0) {
    errors.push("At least one SKU is required");
  }

  // Validate nugget weight
  if (formData.includesNug && (!formData.nuggetTargetWeight || formData.nuggetTargetWeight <= 0)) {
    errors.push("Nugget target weight must be > 0 when nuggets are included");
  }

  if (!formData.includesNug && formData.nuggetTargetWeight) {
    errors.push("Nugget target weight must be null when nuggets are not included");
  }

  // Validate plant and bird size
  if (!formData.reqPlant || !formData.reqBirdSize) {
    errors.push("Plant and bird size are required");
  }

  return errors;
}
```

## Updating Configurations

### Add New Plant

```bash
curl -X PUT http://localhost:8080/api/config/mix.availablePlants \
  -H "Content-Type: application/json" \
  -d '{"value": "FSP,SS2,VBS,JPS"}'
```

### Add New Bird Size

```bash
curl -X PUT http://localhost:8080/api/config/mix.availableBirdSizes \
  -H "Content-Type: application/json" \
  -d '{"value": "SB,BB,XL"}'
```

### Add New Manufacturing Type

```bash
curl -X PUT http://localhost:8080/api/config/mix.availableMfgTypes \
  -H "Content-Type: application/json" \
  -d '{"value": "DSI,DB20,DB30"}'
```

## Database Schema

MIX documents stored in `enumeration_db.mixes` collection:

```javascript
{
  _id: ObjectId,
  skus: {
    "123": "A",
    "456": "B"
  },
  includesFDS: Boolean,
  includesRTL: Boolean,
  includesNug: Boolean,
  nuggetTargetWeight: Number | null,
  numFillets: Number,
  filletWeight: Number,
  mfgType: String, // "DSI" or "DB20"
  cutStrategyID: String,
  beltSpeed: Number,
  reqPlant: String,
  reqBirdSize: String,
  skuSetKey: String // Internal uniqueness key
}
```

### Indexes

```javascript
// Unique constraint: one mix per (mfgType, skuSetKey)
db.mixes.createIndex(
  { mfgType: 1, skuSetKey: 1 },
  { unique: true }
);

// Search indexes
db.mixes.createIndex({ mfgType: 1 });
db.mixes.createIndex({ includesFDS: 1 });
db.mixes.createIndex({ includesRTL: 1 });
db.mixes.createIndex({ includesNug: 1 });
db.mixes.createIndex({ cutStrategyID: 1 });
```


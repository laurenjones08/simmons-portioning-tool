# MIX Configuration Guide

## Overview

The MIX model now includes `reqPlant` and `reqBirdSize` as required fields. The available options for these fields, as well as manufacturing line types, are stored centrally in the **global-config-api** for easy management without code changes.

## MIX Model Fields

### Required Fields

- **reqPlant**: Production plant code (e.g., "FSP", "GSP", "HSP")
- **reqBirdSize**: Bird size classification (e.g., "SB", "MB", "LB")
- **mfgType**: Manufacturing line type (enum: "DSI" or "DB20")

### Example MIX Payload

```json
{
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

## Global Configuration

The available options are stored in `global_config` collection in `config_db`:

### Configuration Keys

| Key | Type | Default Value | Description |
|-----|------|---------------|-------------|
| `mix.availablePlants` | string | `"FSP,SS2,VBS"` | Comma-separated list of available production plants |
| `mix.availableBirdSizes` | string | `"SB,BB"` | Comma-separated list of available bird sizes |
| `mix.availableMfgTypes` | string | `"DSI,DB20"` | Comma-separated list of available manufacturing line types |

### Accessing Configuration via API

```bash
# Get available plants
curl http://localhost:8080/api/config/mix.availablePlants

# Get available bird sizes
curl http://localhost:8080/api/config/mix.availableBirdSizes

# Get available manufacturing types
curl http://localhost:8080/api/config/mix.availableMfgTypes
```

### Updating Configuration

To add a new plant, bird size, or manufacturing type:

```bash
# Add a new plant option
curl -X PUT http://localhost:8080/api/config/mix.availablePlants \
  -H "Content-Type: application/json" \
  -d '{"value": "FSP,VBS,SS2"}'

# Add a new bird size
curl -X PUT http://localhost:8080/api/config/mix.availableBirdSizes \
  -H "Content-Type: application/json" \
  -d '{"value": "SB,BB"}'

# Add a new manufacturing type
curl -X PUT http://localhost:8080/api/config/mix.availableMfgTypes \
  -H "Content-Type: application/json" \
  -d '{"value": "DSI,DB20"}'
```

## MongoDB Schema

The `mixes` collection schema includes:

```javascript
{
  reqPlant: {
    bsonType: "string",
    description: "Required production plant for this mix"
  },
  reqBirdSize: {
    bsonType: "string",
    description: "Required bird size for this mix"
  }
}
```

## Frontend Integration

When building forms for mix creation/editing, fetch the available options from the global config:

```javascript
// Fetch dropdown options
const plants = await fetch('/api/config/mix.availablePlants')
  .then(r => r.json())
  .then(config => config.value.split(','));

const birdSizes = await fetch('/api/config/mix.availableBirdSizes')
  .then(r => r.json())
  .then(config => config.value.split(','));

const mfgTypes = await fetch('/api/config/mix.availableMfgTypes')
  .then(r => r.json())
  .then(config => config.value.split(','));

// Populate dropdowns
// plants: ["FSP", "GSP", "HSP"]
// birdSizes: ["SB", "MB", "LB"]
// mfgTypes: ["DSI", "DB20"]
```

## Benefits

1. **Centralized Management**: Change available options without deploying new code
2. **Consistency**: All services reference the same configuration
3. **Flexibility**: Easy to add/remove options as business requirements change
4. **Audit Trail**: Configuration changes are tracked with `updatedAt` timestamps


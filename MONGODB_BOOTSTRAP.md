# MongoDB Bootstrap Summary

## Overview
Successfully implemented automatic MongoDB database and collection bootstrapping for the Simmons Portioning Tool stack.

## What Was Created

### 1. Initialization Script
**File**: `mongodb-init/init-mongo.js`

This JavaScript file automatically runs when MongoDB starts with a fresh data volume. It creates:

#### Enumeration Database (`enumeration_db`)
- **Collection**: `skus`
- **Purpose**: Stores SKU (Stock Keeping Unit) product data
- **Schema Validation**: Moderate validation with warnings
- **Indexes Created** (6 total):
  - `_id_` - Default unique index on trade number
  - `idx_customer_type` - Fast lookups by customer type
  - `idx_product_type` - Fast lookups by product type
  - `idx_prod_plant` - Fast lookups by production plant
  - `idx_customer_product_plant` - Compound index for multi-criteria searches
  - `idx_customer_name` - Fast lookups by customer name

#### Config Database (`config_db`)
- **Collection**: `global_config`
- **Purpose**: Stores system configuration key-value pairs
- **Schema Validation**: Moderate validation with warnings
- **Indexes Created** (3 total):
  - `_id_` - Default unique index on configuration key
  - `idx_value_type` - Fast filtering by value type
  - `idx_updated_at` - Descending sort by update timestamp
- **Default Configuration Values**:
  - `enumeration.defaultMaxTrim`: 2 (int) - Default max trim for SKU selection
  - `enumeration.defaultMinTargetDelta`: 0.5 (float) - Default min target weight delta
  - `system.enableDebugLogging`: false (bool) - Debug logging flag
  - `system.serviceName`: "Simmons Portioning Tool" (string) - Application name

### 2. Documentation
**File**: `mongodb-init/README.md`

Comprehensive documentation covering:
- How the initialization works
- Database and collection details
- Index descriptions and usage
- Reset procedures
- Troubleshooting guide
- Verification commands

### 3. Docker Compose Update
**File**: `docker-compose.yml`

Updated MongoDB service configuration to mount the initialization script:
```yaml
volumes:
  - mongo_data:/data/db
  - ./mongodb-init/init-mongo.js:/docker-entrypoint-initdb.d/init-mongo.js:ro
```

## Verification Results

### ✅ Databases Created
```
- config_db
- enumeration_db
```

### ✅ Collections Created
```
enumeration_db:
  - skus (0 documents, 6 indexes)

config_db:
  - global_config (4 documents, 3 indexes)
```

### ✅ Default Configuration Loaded
```
- enumeration.defaultMaxTrim: 2 (int)
- enumeration.defaultMinTargetDelta: 0.5 (float)
- system.enableDebugLogging: false (bool)
- system.serviceName: Simmons Portioning Tool (string)
```

### ✅ Services Connected
```
- enumeration-api: Connected to enumeration_db ✓
- global-config-api: Connected to config_db ✓
```

## How It Works

1. **First-Time Only**: The initialization script runs ONLY when MongoDB starts with an empty data directory
2. **Automatic Execution**: MongoDB looks for scripts in `/docker-entrypoint-initdb.d/` and executes them
3. **Persistent Setup**: Once initialized, the databases and collections persist in the Docker volume
4. **Schema Validation**: Documents are validated on insert/update with "warn" action (logs warnings but allows operations)

## Common Operations

### Start the Stack (First Time)
```powershell
docker compose up -d
```
The initialization will run automatically.

### Reset Everything (Start Fresh)
```powershell
# Stop and remove all containers and volumes
docker compose down -v

# Start again (initialization will run)
docker compose up -d
```

### Verify Databases
```powershell
# List all databases
docker exec mongodb mongosh -u root -p example --quiet --eval "db.adminCommand('listDatabases').databases.forEach(function(d){print(d.name)})"

# Check enumeration_db
docker exec mongodb mongosh -u root -p example --quiet --eval "db = db.getSiblingDB('enumeration_db'); print('Collections:', db.getCollectionNames()); print('Indexes:', db.skus.getIndexes().length); print('Documents:', db.skus.countDocuments({}));"

# Check config_db
docker exec mongodb mongosh -u root -p example --quiet --eval "db = db.getSiblingDB('config_db'); print('Collections:', db.getCollectionNames()); print('Indexes:', db.global_config.getIndexes().length); print('Documents:', db.global_config.countDocuments({}));"

# View default configuration
docker exec mongodb mongosh -u root -p example --quiet --eval "db = db.getSiblingDB('config_db'); db.global_config.find().forEach(function(doc){print('  - ' + doc.key + ': ' + doc.value + ' (' + doc.valueType + ')')});"
```

### Interactive MongoDB Shell
```powershell
# Connect to MongoDB shell
docker exec -it mongodb mongosh -u root -p example

# Then run commands:
show dbs
use enumeration_db
show collections
db.skus.getIndexes()
db.skus.find()

use config_db
db.global_config.find().pretty()
```

## Schema Validation Details

### SKU Schema
Required fields:
- `_id` (string) - Trade number identifier
- `tradeNumber` (string)
- `customerName` (string)
- `customerType` (string)
- `productType` (string)
- `unitsPerCut` (int, min: 1)
- `prodPlant` (string)
- `minWeight` (double, min: 0)
- `maxWeight` (double, min: 0)
- `targetWeight` (double, min: 0)
- `allowedParts` (array of strings, min items: 1)

### Config Schema
Required fields:
- `_id` (string) - Configuration key
- `key` (string)
- `value` (int | string | double | bool)
- `valueType` (enum: "int", "string", "float", "bool")
- `description` (string)
- `updatedAt` (date)

Optional fields:
- `minValue` (for numeric types)
- `maxValue` (for numeric types)

## Performance Optimizations

### Index Strategy
1. **Single-field indexes** for common filter operations
2. **Compound index** for multi-criteria searches (uses left-to-right matching)
3. **Background creation** to avoid blocking operations
4. **Automatic _id indexes** for primary key lookups (O(1) performance)

### Query Optimization
The compound index `{customerType: 1, productType: 1, prodPlant: 1}` supports:
- Queries on `customerType` alone
- Queries on `customerType` + `productType`
- Queries on all three fields

MongoDB can efficiently use this compound index for any left-prefix.

## Next Steps

### Adding Sample Data (Optional)
To add sample SKU data, uncomment the `db.skus.insertMany([...])` section in `init-mongo.js` and reset the volume.

### Adding More Default Configs
Edit the `db.global_config.insertMany([...])` section in `init-mongo.js` to add more default configuration values.

### Adding New Collections
Add new `db.createCollection()` calls in `init-mongo.js` following the same pattern.

### Adding More Indexes
Add new `db.<collection>.createIndex()` calls for additional search patterns.

## Troubleshooting

### Issue: Databases Not Created
**Solution**: Check MongoDB logs for errors:
```powershell
docker logs mongodb | Select-String -Pattern "init-mongo","Error","Failed"
```

### Issue: Need to Re-run Initialization
**Solution**: Remove the volume and restart:
```powershell
docker compose down -v
docker compose up -d
```

### Issue: Schema Validation Errors
**Solution**: Check validation is set to "warn" mode (logs but doesn't block). Review MongoDB logs:
```powershell
docker logs mongodb --tail 50
```

## Benefits

1. ✅ **Zero Manual Setup** - Databases and collections created automatically
2. ✅ **Consistent Schema** - Schema validation ensures data quality
3. ✅ **Optimized Performance** - Pre-created indexes for common queries
4. ✅ **Default Configuration** - System ready to use immediately
5. ✅ **Version Controlled** - Initialization script tracked in Git
6. ✅ **Documented** - Comprehensive README for maintenance
7. ✅ **Reproducible** - Same setup on any environment

## Files Modified/Created

```
mongodb-init/
├── init-mongo.js          # Initialization script (NEW)
└── README.md              # Documentation (NEW)

docker-compose.yml         # Updated volume mount (MODIFIED)
```

## Success Criteria ✅

- [x] enumeration_db created automatically
- [x] config_db created automatically
- [x] skus collection created with schema validation
- [x] global_config collection created with schema validation
- [x] All indexes created successfully
- [x] Default configuration values loaded
- [x] enumeration-api connects successfully
- [x] global-config-api connects successfully
- [x] Initialization runs on fresh start
- [x] Initialization script documented
- [x] Reset procedure documented


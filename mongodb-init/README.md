# MongoDB Initialization

This directory contains scripts for automatically bootstrapping MongoDB databases and collections when the MongoDB container starts.

## Overview

The `init-mongo.js` script automatically creates:

### 1. Enumeration Database (`enumeration_db`)
- **Collection**: `skus`
- **Purpose**: Stores Stock Keeping Unit (SKU) data for product enumeration
- **Indexes**:
  - `idx_customer_type` - Fast filtering by customer type
  - `idx_product_type` - Fast filtering by product type
  - `idx_prod_plant` - Fast filtering by production plant
  - `idx_customer_product_plant` - Compound index for multi-criteria searches
  - `idx_customer_name` - Fast filtering by customer name

### 2. Config Database (`config_db`)
- **Collection**: `global_config`
- **Purpose**: Stores global configuration key-value pairs
- **Indexes**:
  - `idx_value_type` - Fast filtering by value type
  - `idx_updated_at` - Sorting by last update timestamp
- **Optional Snapshot Restore**:
  - If `mongodb-bootstrap.archive.gz` exists in this folder, Mongo restores that archive first
  - After a restore, `init-mongo.js` detects the restore marker and skips the default seed path
- **Default Configuration Values**:
  - `enumeration.defaultMaxTrim` - Default max trim (int: 15)
  - `enumeration.defaultMinTargetDelta` - Default min target delta (float: 0.5)
  - `enumeration.bucketWeightTolerancePct` - Bucket fit tolerance percent (float: 0.0)
  - `enumeration.fdsValueCoefficient` - FDS value coefficient used in mix scoring (float: 0.0)
  - `enumeration.rtlValueCoefficient` - RTL value coefficient used in mix scoring (float: 0.0)
  - `enumeration.trimValueCoefficient` - Trim value coefficient used in mix scoring (float: 0.0)
  - `system.enableDebugLogging` - Debug logging flag (bool: false)
  - `system.serviceName` - Application name (string: "Simmons Portioning Tool")

## How It Works

1. When the MongoDB container starts for the **first time**, it looks for scripts in `/docker-entrypoint-initdb.d/`
2. The `00-restore-bootstrap.sh` script checks for `mongodb-bootstrap.archive.gz`
3. If the archive exists, MongoDB restores it before running the JS bootstrap
4. The `init-mongo.js` script is mounted to this directory via Docker Compose
5. MongoDB executes the JS script automatically using `mongosh`
6. If no archive exists, the script creates databases, collections, indexes, and default data
7. On subsequent container starts, the scripts are **not re-executed** (MongoDB tracks initialization)

## Important Notes

### First-Time Initialization Only
The initialization script only runs when MongoDB starts with an **empty data directory**. If you need to re-run the initialization:

```powershell
# Stop and remove containers
docker compose down

# Remove the MongoDB data volume
docker volume rm simmons-portioning-tool_mongo_data

# Start containers again (initialization will run)
docker compose up -d
```

### Modifying the Initialization Script

If you modify `init-mongo.js`, you must reset the MongoDB data volume for changes to take effect:

```powershell
# Option 1: Full reset (removes all data)
docker compose down -v
docker compose up -d

# Option 2: Selective volume removal
docker compose down
docker volume rm simmons-portioning-tool_mongo_data
docker compose up -d
```

### Adding Sample Data

The script includes commented-out sample SKU data. To enable it:
1. Uncomment the `db.skus.insertMany([...])` section in `init-mongo.js`
2. Reset the MongoDB volume (see above)
3. Restart the containers

### Capturing the Current Database State

To snapshot the live MongoDB contents into the repo:

```powershell
.\scripts\capture-mongodb-state.ps1
```

That writes `mongodb-init/mongodb-bootstrap.archive.gz`, which the startup restore hook will apply automatically on the next fresh container initialization.

## Verification

After the containers start, you can verify the initialization:

```powershell
# Connect to MongoDB container
docker exec -it mongodb mongosh -u root -p example

# Check databases
show dbs

# Check enumeration_db
use enumeration_db
show collections
db.skus.getIndexes()
db.skus.countDocuments({})

# Check config_db
use config_db
show collections
db.global_config.getIndexes()
db.global_config.find().pretty()
```

## Performance Considerations

### Indexes
All indexes are created with `background: true` to avoid blocking other operations during creation.

### Index Usage
The compound index `idx_customer_product_plant` is optimized for common search patterns:
- Searching by `customerType` alone
- Searching by `customerType` and `productType`
- Searching by all three fields: `customerType`, `productType`, and `prodPlant`

MongoDB can use this compound index efficiently for any left-to-right subset of the indexed fields.

## Troubleshooting

### Script Didn't Run
**Symptom**: Databases or collections are missing after container startup.

**Solution**:
1. Check MongoDB logs: `docker logs mongodb`
2. Ensure the volume mount is correct in `docker-compose.yml`
3. Verify the script has no syntax errors
4. Reset the data volume and restart


### Permission Issues
**Symptom**: Cannot read initialization script.

**Solution**:
1. Ensure the script file has read permissions
2. Check the volume mount in `docker-compose.yml` includes `:ro` (read-only)
3. On Windows, ensure Docker Desktop has access to the project directory


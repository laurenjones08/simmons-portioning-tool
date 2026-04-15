# MongoDB Reinitialization Guide

## Overview

The MongoDB initialization script (`mongodb-init/init-mongo.js`) only runs automatically when MongoDB starts with an **empty database**. This guide explains how to force re-execution of the bootstrap script.

If `mongodb-init/mongodb-bootstrap.archive.gz` exists, the container bootstrap now restores that archive first and then skips the default seed path. That makes the repo snapshot the authoritative startup state.

## Methods to Reinitialize MongoDB

### Method 1: PowerShell Script (Recommended for Windows)

```powershell
.\scripts\reinit-mongodb.ps1
```

**What it does:**
- Drops `enumeration_db` and `config_db` databases
- Re-runs the initialization script
- Recreates all collections, indexes, and default configuration data
- Interactive confirmation prompt to prevent accidents

**Advantages:**
- ✅ Keeps container running (no downtime)
- ✅ Preserves volumes
- ✅ Fast execution (no container rebuild)
- ✅ Safe with confirmation prompt

---

### Method 2: Bash Script (Linux/Mac)

```bash
./scripts/reinit-mongodb.sh
```

Same functionality as the PowerShell script, but for Unix-based systems.

---

### Method 3: Docker Compose Volume Reset

```bash
# Stop all services and DELETE the MongoDB volume
docker compose down -v

# Restart (initialization runs automatically on empty database)
docker compose up -d --build
```

**What it does:**
- Completely removes the MongoDB data volume
- Destroys all data (cannot be recovered)
- Next startup triggers automatic initialization

**Advantages:**
- ✅ Complete clean slate
- ✅ No manual script execution needed

**Disadvantages:**
- ⚠️ Requires container restart (downtime)
- ⚠️ Permanently deletes ALL MongoDB data
- ⚠️ Longer execution time (container rebuild)

---

### Method 4: Manual MongoDB Commands

```bash
# Connect to MongoDB
docker exec -it mongodb mongosh -u root -p example

# Drop databases manually
use enumeration_db
db.dropDatabase()

use config_db
db.dropDatabase()

exit

# Re-run init script
Get-Content mongodb-init/init-mongo.js | docker exec -i mongodb mongosh -u root -p example --quiet
```

**Use when:**
- You need fine-grained control
- You want to drop only specific collections
- Debugging initialization issues

### Method 5: Capture the Current State Into the Repo

```powershell
.\scripts\capture-mongodb-state.ps1
```

This exports the live MongoDB contents into `mongodb-init/mongodb-bootstrap.archive.gz`. On the next fresh container start, the bootstrap restore hook applies that archive automatically.

---

## What Gets Initialized

### `enumeration_db` Database

**Collections:**
1. **`skus`** - Stock Keeping Unit data
   - 6 indexes (customer type, product type, plant, compound, customer name)

2. **`mixes`** - Mix combinations for portioning
   - 8 indexes (unique compound, mfg type, flags, cut strategy)

### `config_db` Database

**Collections:**
1. **`global_config`** - System configuration key-value store
   - 2 indexes (value type, updated timestamp)
   - Default configuration values:
     - `enumeration.defaultMaxTrim`
     - `enumeration.defaultMinTargetDelta`
      - `enumeration.bucketWeightTolerancePct` = `0.0`
      - `enumeration.fdsValueCoefficient` = `0.0`
      - `enumeration.rtlValueCoefficient` = `0.0`
      - `enumeration.trimValueCoefficient` = `0.0`
     - `system.enableDebugLogging`
     - `system.serviceName`
     - `mix.availablePlants` = `"FSP,SS2,VBS"`
     - `mix.availableBirdSizes` = `"SB,BB"`
      - `mix.availableMfgTypes` = `"DSI888,DSI884,DB20"`

---

## Verification

After reinitialization, verify the setup:

```bash
# Check databases exist
docker exec mongodb mongosh -u root -p example --eval "db.adminCommand('listDatabases')"

# Check enumeration_db collections
docker exec mongodb mongosh -u root -p example --eval "db.getSiblingDB('enumeration_db').getCollectionNames()"

# Check config_db default values
docker exec mongodb mongosh -u root -p example --eval "db.getSiblingDB('config_db').global_config.countDocuments({})"

# Should return 11 config documents
```

---

## Troubleshooting

### Script fails with "container not found"

**Solution:** Ensure MongoDB container is running:
```bash
docker compose up -d mongodb
```

---

### Permission denied on scripts

**Linux/Mac:**
```bash
chmod +x scripts/reinit-mongodb.sh
```

**Windows:** PowerShell execution policy:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### Initialization script not running

The init script only runs when:
1. The database is completely empty
2. Container is starting for the first time
3. No existing `/data/db` files

**Fix:** Use one of the reinitialization methods above to force execution.

---

## Best Practices

1. **Development:** Use `reinit-mongodb.ps1` script for quick resets
2. **Testing:** Use `docker compose down -v` for complete clean slate
3. **Production:** Never use these methods (use proper migrations instead)
4. **Backup first:** Before reinitializing, export data if needed:
   ```bash
   docker exec mongodb mongodump -u root -p example --out=/backup
   ```


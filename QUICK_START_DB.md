# Quick Start: MongoDB Database Management

## Initial Setup

1. **Start the stack:**
   ```bash
   docker compose up -d --build
   ```

2. **The initialization script runs automatically** on first startup when the database is empty.

---

## Force Reinitialization

Need to reset your database? Use the reinit script:

### Windows
```powershell
.\scripts\reinit-mongodb.ps1
```

### Linux/Mac
```bash
./scripts/reinit-mongodb.sh
```

### What happens:
1. ✅ Drops `enumeration_db` and `config_db`
2. ✅ Re-creates all collections
3. ✅ Rebuilds all indexes
4. ✅ Restores default configuration values

**Safety:** Interactive prompt asks for confirmation before dropping data.

---

## Nuclear Option

Complete reset (destroys volume):

```bash
docker compose down -v
docker compose up -d --build
```

⚠️ **WARNING:** This permanently deletes ALL data.

---

## Verify Database State

```bash
# List all databases
docker exec mongodb mongosh -u root -p example --eval "db.adminCommand('listDatabases')"

# Count SKUs
docker exec mongodb mongosh -u root -p example --eval "db.getSiblingDB('enumeration_db').skus.countDocuments({})"

# Count Mixes
docker exec mongodb mongosh -u root -p example --eval "db.getSiblingDB('enumeration_db').mixes.countDocuments({})"

# Count Config entries (should be 7)
docker exec mongodb mongosh -u root -p example --eval "db.getSiblingDB('config_db').global_config.countDocuments({})"

# List config keys
docker exec mongodb mongosh -u root -p example --eval "db.getSiblingDB('config_db').global_config.find({}, {key:1, value:1, _id:0}).pretty()"
```

---

## Access Configuration via API

Once services are running:

```bash
# Get all config
curl http://localhost:8080/api/config/

# Get specific config
curl http://localhost:8080/api/config/mix.availablePlants
curl http://localhost:8080/api/config/mix.availableBirdSizes
curl http://localhost:8080/api/config/mix.availableMfgTypes
```

---

## Common Scenarios

### Scenario 1: "I changed init-mongo.js and want to apply changes"
```powershell
.\scripts\reinit-mongodb.ps1
```

### Scenario 2: "I want to start completely fresh"
```bash
docker compose down -v
docker compose up -d --build
```

### Scenario 3: "I just want to add new config without dropping everything"
```bash
# Connect to MongoDB
docker exec -it mongodb mongosh -u root -p example

# Add config manually
use config_db
db.global_config.insertOne({
  "_id": "new.configKey",
  "key": "new.configKey",
  "value": "someValue",
  "valueType": "string",
  "description": "New configuration",
  "updatedAt": new Date()
})
```

---

## Documentation

- **Full reinitialization guide:** `MONGODB_REINITIALIZATION_GUIDE.md`
- **MIX configuration details:** `MIX_CONFIGURATION_GUIDE.md`
- **Add new microservices:** `MICROSERVICE_API_ONBOARDING.md`


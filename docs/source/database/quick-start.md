# Database Quick Start

Fast reference for common database tasks.

## Quick Commands

### Check Database Status

```bash
# Test MongoDB connection
docker exec mongodb mongosh -u root -p example --eval "db.runCommand({ ping: 1 })"

# List all databases
docker exec mongodb mongosh -u root -p example --eval "db.adminCommand('listDatabases')"
```

### View Data

```bash
# Count SKUs
docker exec mongodb mongosh -u root -p example --eval "db.getSiblingDB('enumeration_db').skus.countDocuments({})"

# Count MIXes
docker exec mongodb mongosh -u root -p example --eval "db.getSiblingDB('enumeration_db').mixes.countDocuments({})"

# List all config keys
docker exec mongodb mongosh -u root -p example --eval "db.getSiblingDB('config_db').global_config.find({}, {key:1, value:1, _id:0}).pretty()"
```

### Reset Database

**Windows:**
```powershell
.\scripts\reinit-mongodb.ps1
```

**Linux/Mac:**
```bash
./scripts/reinit-mongodb.sh
```

## Database Reinitialization

The `/mixes` collection and all indexes are created during the initialization script.

**Auto-initialization** occurs when:
- MongoDB starts with no databases
- First docker-compose up after database deletion

**Manual reinitialization** options:

1. **Quick Reset (keeps container running):**
   ```powershell
   .\scripts\reinit-mongodb.ps1
   ```

2. **Complete Reset (removes volume):**
   ```bash
   docker compose down -v
   docker compose up -d --build
   ```

## Verify Database Setup

After initialization or restart, verify:

```bash
# Check collections exist
docker exec mongodb mongosh -u root -p example --eval "
  db.getSiblingDB('enumeration_db').getCollectionNames();
  db.getSiblingDB('config_db').getCollectionNames();
"

# Check indexes
docker exec mongodb mongosh -u root -p example --eval "
  db.getSiblingDB('enumeration_db').mixes.getIndexes()
"

# Check config defaults
docker exec mongodb mongosh -u root -p example --eval "
  db.getSiblingDB('config_db').global_config.find().pretty()
"
```

## Common Issues

| Issue | Fix |
|-------|-----|
| "Connection refused" | Check `docker compose ps`, ensure mongodb is running |
| "database not found" | Run reinitialization script |
| "index already exists" | Normal on restart; indexes are idempotent |
| "permission denied" | Check MongoDB credentials in docker-compose.yml |

See [Reinitialization Guide](reinitialization.md) for more details.


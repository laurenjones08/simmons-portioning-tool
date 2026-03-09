# Running the Stack

Complete guide to starting, managing, and stopping the Simmons Portioning Tool stack.

## Starting the Stack

### Development Mode

Start with live code reloading:

```bash
docker compose up -d --build
```

**What starts:**
- MongoDB (Port 27017, internal)
- Enumeration API (Port 8000, internal)
- Global Config API (Port 8001, internal)
- API Gateway (Port 8080, public)
- Jaeger (Port 16686, public)

### Production Mode

Build and start with optimizations:

```bash
# Build with no cache
docker compose build --no-cache

# Start services
docker compose up -d
```

## Checking Service Status

### View All Containers

```bash
docker compose ps
```

**Output:**
```
NAME                COMMAND                  SERVICE             STATUS       PORTS
mongodb             "docker-entrypoint.sh…" mongodb             Up 10s       0.0.0.0:27017->27017/tcp
enumeration-api     "python -m uvicorn ma…" enumeration-api     Up 8s
global-config-api   "python -m uvicorn ma…" global-config-api   Up 8s
api-gateway         "/docker-entrypoint.s…" api-gateway         Up 5s        0.0.0.0:8080->80/tcp
jaeger              "/go/bin/all-in-one-l…" jaeger              Up 2s        0.0.0.0:16686->16686/tcp
```

### Check Specific Service

```bash
# Check MongoDB status
docker compose exec mongodb mongosh -u root -p example --eval "db.adminCommand('ping')"

# Check Enumeration API
curl http://localhost:8080/api/enumeration/health

# Check Config API
curl http://localhost:8080/api/config/health

# Check Gateway
curl http://localhost:8080/health
```

## Viewing Logs

### View All Logs

```bash
# View recent logs from all services
docker compose logs --tail=50

# Follow logs in real-time
docker compose logs -f

# Follow logs from specific service
docker compose logs -f enumeration-api
```

### Common Log Patterns

**Service Started Successfully:**
```
enumeration-api      | INFO:     Uvicorn running on http://0.0.0.0:8000
enumeration-api      | INFO:     Application startup complete
```

**Database Connected:**
```
enumeration-api      | Connected to MongoDB
enumeration-api      | Database: enumeration_db
```

**Initialization Running:**
```
mongodb              | [initandlisten] Database init script starting
mongodb              | ✓ Database "enumeration_db" created
mongodb              | ✓ Database "config_db" created
```

## Testing the Stack

### Test API Gateway

```bash
# Health check
curl http://localhost:8080/health

# List routes
curl http://localhost:8080/
```

### Test Enumeration API

```bash
# Check health
curl http://localhost:8080/api/enumeration/health

# Get OpenAPI schema
curl http://localhost:8080/api/enumeration/openapi.json | jq

# List all SKUs (requires sample data)
curl http://localhost:8080/api/enumeration/skus/search \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Test Global Config API

```bash
# Check health
curl http://localhost:8080/api/config/health

# Get config key
curl http://localhost:8080/api/config/mix.availablePlants

# List all config
curl http://localhost:8080/api/config/
```

## Interactive API Documentation

Once running, access interactive docs:

**Enumeration API:**
- Swagger UI: `http://localhost:8080/api/enumeration/docs`
- ReDoc: `http://localhost:8080/api/enumeration/redoc`

**Global Config API:**
- Swagger UI: `http://localhost:8080/api/config/docs`
- ReDoc: `http://localhost:8080/api/config/redoc`

## Database Initialization

### First Startup (Automatic)

On first startup with empty database, the initialization script automatically:
1. Creates `enumeration_db` and `config_db`
2. Sets up `skus` and `mixes` collections
3. Creates optimized indexes
4. Populates default configuration

### Manual Reinitialization

To reset database to clean state:

**Windows:**
```powershell
.\scripts\reinit-mongodb.ps1
```

**Linux/Mac:**
```bash
./scripts/reinit-mongodb.sh
```

### Verify Initialization

```bash
# Check databases exist
docker exec mongodb mongosh -u root -p example --eval "db.adminCommand('listDatabases')"

# Check collections
docker exec mongodb mongosh -u root -p example --eval "db.getSiblingDB('enumeration_db').getCollectionNames()"

# Count documents
docker exec mongodb mongosh -u root -p example --eval "db.getSiblingDB('config_db').global_config.countDocuments({})"
```

## Stopping the Stack

### Stop All Services

```bash
docker compose stop
```

Services stop but containers and volumes remain.

### Stop and Remove Containers

```bash
docker compose down
```

Containers are removed but volumes (data) persist.

### Full Reset (⚠️ Deletes All Data)

```bash
docker compose down -v
```

Removes containers, networks, AND volumes. Database will reinitialize on next startup.

## Resource Management

### View Resource Usage

```bash
# View current usage
docker stats

# View disk usage
docker system df
```

### Free Up Space

```bash
# Remove unused containers
docker container prune

# Remove unused images
docker image prune -a

# Remove unused volumes (⚠️ removes data!)
docker volume prune

# Complete cleanup
docker system prune -a
```

## Rebuilding Services

### Rebuild Single Service

```bash
docker compose build enumeration-api
docker compose up -d enumeration-api
```

### Rebuild All Services

```bash
docker compose build --no-cache
docker compose up -d
```

## Environment Variables

Override settings via environment:

```bash
# Set on command line
MONGODB_URL=mongodb://root:pass@mongodb:27017 docker compose up

# Or create .env file
cat > .env << EOF
MONGODB_URL=mongodb://root:example@mongodb:27017
ENUMERATION_DB=enumeration_db
CONFIG_DB=config_db
EOF

docker compose up -d
```

## Debugging

### Execute Commands in Container

```bash
# Run shell in enumeration-api
docker compose exec enumeration-api /bin/bash

# Run Python command
docker compose exec enumeration-api python -c "import sys; print(sys.version)"

# Check environment variables
docker compose exec enumeration-api env | grep MONGO
```

### View Container Logs

```bash
# Last 100 lines
docker compose logs --tail=100

# With timestamps
docker compose logs --timestamps

# Show errors only
docker compose logs | grep -i error
```

### Access MongoDB Directly

```bash
# Start mongosh shell
docker compose exec mongodb mongosh -u root -p example

# List databases
db.adminCommand('listDatabases')

# Switch database
use enumeration_db

# Query collections
db.skus.find()
db.mixes.find()
```

## Next Steps

- 📖 Learn the [Architecture](../architecture/overview.md)
- 🔧 Explore [API Documentation](../api/enumeration/overview.md)
- 💾 Understand [Database Setup](../database/schema.md)
- 🚀 [Add a New Microservice](../development/adding-microservices.md)

## Troubleshooting

See [Troubleshooting Guide](../operations/troubleshooting.md) for common issues.


# Quick Start Guide

Get the Simmons Portioning Tool up and running in 5 minutes!

## Prerequisites

- Docker and Docker Compose installed
- Git (to clone the repository)
- 4GB+ free disk space

## Quick Start (5 minutes)

### 1. Clone and Navigate

```bash
git clone https://github.com/your-org/simmons-portioning-tool.git
cd simmons-portioning-tool
```

### 2. Start the Stack

```bash
docker compose up -d --build
```

This starts:
- ✅ MongoDB database
- ✅ Enumeration API (SKU management)
- ✅ Global Config API (system settings)
- ✅ API Gateway (unified entry point)
- ✅ Jaeger (distributed tracing)

### 3. Verify Everything is Running

```bash
# Check container status
docker compose ps

# Test the gateway health check
curl http://localhost:8080/health
```

Expected output:
```json
{"status":"healthy","service":"api-gateway"}
```

### 4. Access the Documentation

| Component | URL |
|-----------|-----|
| **This Documentation** | `http://localhost:3000` |
| **Enumeration API Docs** | `http://localhost:8080/api/enumeration/docs` |
| **Global Config API Docs** | `http://localhost:8080/api/config/docs` |
| **Jaeger Tracing** | `http://localhost:16686` |

---

## Common First Steps

### Create a MIX Configuration

```bash
curl -X POST http://localhost:8080/api/enumeration/mixes \
  -H "Content-Type: application/json" \
  -d '{
    "skus": {"123": "A", "456": "B"},
    "includesFDS": true,
    "includesRTL": false,
    "includesNug": true,
    "nuggetTargetWeight": 15.5,
    "numFillets": 2,
    "filletWeight": 12.75,
    "mfgType": "DSI",
    "cutStrategyID": "strategy-1",
    "beltSpeed": 1.2,
    "reqPlant": "FSP",
    "reqBirdSize": "SB"
  }'
```

### Check Available Configuration Options

```bash
curl http://localhost:8080/api/config/mix.availablePlants
curl http://localhost:8080/api/config/mix.availableBirdSizes
curl http://localhost:8080/api/config/mix.availableMfgTypes
```

### Reset the Database

```powershell
# Windows
.\scripts\reinit-mongodb.ps1

# Linux/Mac
./scripts/reinit-mongodb.sh
```

---

## Stop the Stack

```bash
docker compose down
```

To remove all data:
```bash
docker compose down -v
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port already in use | Change port in `docker-compose.yml` |
| MongoDB connection failed | Check MongoDB logs: `docker compose logs mongodb` |
| API returns 502 | Wait 10s and retry, services may still be starting |

For more help, see [Troubleshooting](../operations/troubleshooting.md).

---

## Next Steps

- 📖 Read the [System Architecture](../architecture/overview.md)
- 🔧 Learn [how to add new services](../development/adding-microservices.md)
- 💾 Understand [database setup](../database/schema.md)
- 🚀 Review [API documentation](../api/enumeration/overview.md)


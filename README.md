# Portioning App (Streamlit)

[![Enumeration API Tests](https://github.com/laurenjones08/simmons-portioning-tool/actions/workflows/enumeration-tests.yml/badge.svg)](https://github.com/laurenjones08/simmons-portioning-tool/actions/workflows/enumeration-tests.yml)
[![Global Config API Tests](https://github.com/laurenjones08/simmons-portioning-tool/actions/workflows/global-config-tests.yml/badge.svg)](https://github.com/laurenjones08/simmons-portioning-tool/actions/workflows/global-config-tests.yml)
[![Stack Integration](https://github.com/laurenjones08/simmons-portioning-tool/actions/workflows/stack-integration.yml/badge.svg)](https://github.com/laurenjones08/simmons-portioning-tool/actions/workflows/stack-integration.yml)

## 📚 Quick Links

- **📖 [Documentation Portal](http://localhost:3000)** - Complete documentation site (MkDocs)
- **🚀 [Database Quick Start](QUICK_START_DB.md)** - Common DB operations
- **🔧 [MIX Configuration](MIX_CONFIGURATION_GUIDE.md)** - Configure plants, bird sizes, mfg types
- **📋 [MongoDB Reinitialization](MONGODB_REINITIALIZATION_GUIDE.md)** - Reset database
- **➕ [Add New Microservices](MICROSERVICE_API_ONBOARDING.md)** - Service onboarding guide

---

## Inputs
- Enumeration engine expects a sheet/CSV with columns like:
  - TradeNumber, CustomerType, TargetWeight, BirdSize, ProdPlant, AllowedParts

## Unified API Gateway (one roof)

The microservice APIs are available through one public endpoint:

- Base URL: `http://localhost:8080`
- Enumeration API via gateway: `http://localhost:8080/api/enumeration/*`
- Global Config API via gateway: `http://localhost:8080/api/config/*`
- Gateway health: `http://localhost:8080/health`

Start the stack:

```bash
docker compose up -d --build
```

## Reinitialize MongoDB Database

To reset the MongoDB databases to a clean state and re-run the bootstrap script:

**Windows (PowerShell):**
```powershell
.\scripts\reinit-mongodb.ps1
```

**Linux/Mac (Bash):**
```bash
./scripts/reinit-mongodb.sh
```

This will:
1. Drop `enumeration_db` and `config_db` databases
2. Re-run the initialization script to recreate collections, indexes, and default data
3. Confirm completion with a summary

**Alternative (manual):**
```bash
# Stop services and remove the MongoDB volume
docker compose down -v

# Restart (initialization script runs automatically on empty database)
docker compose up -d --build
```

See `MICROSERVICE_API_ONBOARDING.md` for how to add new microservice APIs to this setup.

## Long-Running Enumeration Worker

Use the `enumeration-worker` container for long-running staged enumeration across SKU combinations of size 1 through 4.

- Stage 1: all 1-SKU combos
- Stage 2: all 2-SKU combos
- Stage 3: all 3-SKU combos
- Stage 4: all 4-SKU combos

The worker stores run progress checkpoints in `enumeration_runs` and outputs in `enumeration_results`.
If the worker is restarted with the same `ENUMERATION_RUN_ID`, it resumes from the last stage checkpoint.

Run it once:

```powershell
docker compose run --rm enumeration-worker
```

Override run settings:

```powershell
docker compose run --rm `
  -e ENUMERATION_RUN_ID=run-2026-03-13 `
  -e ENUMERATION_BATCH_SIZE=500 `
  -e ENUMERATION_MAX_COMBINATION_SIZE=4 `
  -e ENUMERATION_SKUS=50624,50625,50626 `
  enumeration-worker
```

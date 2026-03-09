# Portioning App (Streamlit)

## 📚 Quick Links

- **📖 [Documentation Portal](http://localhost:3000)** - Complete documentation site (MkDocs)
- **🚀 [Database Quick Start](QUICK_START_DB.md)** - Common DB operations
- **🔧 [MIX Configuration](MIX_CONFIGURATION_GUIDE.md)** - Configure plants, bird sizes, mfg types
- **📋 [MongoDB Reinitialization](MONGODB_REINITIALIZATION_GUIDE.md)** - Reset database
- **➕ [Add New Microservices](MICROSERVICE_API_ONBOARDING.md)** - Service onboarding guide

---

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

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

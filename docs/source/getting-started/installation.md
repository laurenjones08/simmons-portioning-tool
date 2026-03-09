# Installation Guide

Complete step-by-step installation instructions for the Simmons Portioning Tool.

## System Requirements

### Minimum Requirements
- CPU: 2 cores
- RAM: 4GB
- Storage: 5GB free space
- OS: Windows 10+, macOS 10.14+, or Linux

### Recommended Requirements
- CPU: 4 cores
- RAM: 8GB
- Storage: 20GB free space

## Prerequisites

### 1. Install Docker and Docker Compose

**Windows:**
- Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
- Install and enable WSL 2 backend
- Verify: `docker --version`

**Mac:**
- Download [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop)
- Verify: `docker --version`

**Linux (Ubuntu/Debian):**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify
docker --version
docker compose --version
```

### 2. Install Git

**Windows:**
- Download [Git for Windows](https://git-scm.com/download/win)
- Or via Chocolatey: `choco install git`

**Mac:**
- Via Homebrew: `brew install git`

**Linux:**
```bash
sudo apt-get install git
```

### 3. Clone the Repository

```bash
git clone https://github.com/your-org/simmons-portioning-tool.git
cd simmons-portioning-tool
```

## Build the Stack

### 1. Configure Environment Variables (Optional)

Create `.env` file in the project root:

```env
# MongoDB
MONGO_INITDB_ROOT_USERNAME=root
MONGO_INITDB_ROOT_PASSWORD=example
MONGODB_URL=mongodb://root:example@mongodb:27017

# API Services
ENUMERATION_DB=enumeration_db
CONFIG_DB=config_db

# Jaeger Tracing
JAEGER_AGENT_HOST=jaeger
JAEGER_AGENT_PORT=6831
```

### 2. Build and Start Services

```bash
# Build all services
docker compose build

# Start all services
docker compose up -d

# View logs
docker compose logs -f
```

### 3. Verify Installation

```bash
# Check running containers
docker compose ps

# Test health endpoint
curl http://localhost:8080/health

# Check specific service health
docker compose logs enumeration-api | grep -i "started"
docker compose logs global-config-api | grep -i "started"
```

## Initialization

The database initializes automatically on first startup. This includes:
- Creating `enumeration_db` and `config_db` databases
- Setting up collections and indexes
- Populating default configuration values

### Manual Reinitialization

If you need to reset the database:

**Windows (PowerShell):**
```powershell
.\scripts\reinit-mongodb.ps1
```

**Linux/Mac (Bash):**
```bash
./scripts/reinit-mongodb.sh
```

## Access Points

Once running, access via:

| Service | URL | Purpose |
|---------|-----|---------|
| **Documentation** | `http://localhost:3000` | This documentation |
| **API Gateway** | `http://localhost:8080` | Unified API entry |
| **Enumeration API Docs** | `http://localhost:8080/api/enumeration/docs` | Interactive Swagger UI |
| **Config API Docs** | `http://localhost:8080/api/config/docs` | Interactive Swagger UI |
| **Jaeger Tracing** | `http://localhost:16686` | Distributed tracing UI |
| **MongoDB** | `localhost:27017` | Database connection |

## Troubleshooting Installation

### Issue: Ports Already in Use

**Error:** `Bind for 0.0.0.0:8080 failed: port is already allocated`

**Solution:**
Edit `docker-compose.yml`:
```yaml
api-gateway:
  ports:
    - "8888:80"  # Change 8080 to 8888
```

Then restart:
```bash
docker compose down
docker compose up -d
```

### Issue: Docker Daemon Not Running

**Error:** `Cannot connect to Docker daemon`

**Solution:**
```bash
# Windows
# Restart Docker Desktop via system tray

# Linux
sudo systemctl restart docker

# Mac
# Restart Docker Desktop
```

### Issue: Insufficient Disk Space

**Solution:**
```bash
# Clean up old images
docker image prune -a

# Clean up unused volumes
docker volume prune

# Check disk usage
docker system df
```

### Issue: Build Fails with Dependency Error

**Error:** `No matching distribution found for package-name==version`

**Solution:**
```bash
# Clear Docker cache and rebuild
docker compose build --no-cache

# Update requirements.txt versions if needed
```

## Verification Checklist

- [ ] All containers are running: `docker compose ps`
- [ ] Gateway health check passes: `curl http://localhost:8080/health`
- [ ] MongoDB is responsive: `docker compose exec mongodb mongosh -u root -p example`
- [ ] Configuration is loaded: `curl http://localhost:8080/api/config/mix.availablePlants`
- [ ] Documentation is accessible: Visit `http://localhost:3000`

## Next Steps

1. [Run the Stack](running-the-stack.md)
2. [Understand the Architecture](../architecture/overview.md)
3. [Explore API Documentation](../api/enumeration/overview.md)
4. [Configure Mix Settings](../features/mix/configuration.md)

## Getting Help

- 📖 Check [Troubleshooting Guide](../operations/troubleshooting.md)
- 🐛 Review logs: `docker compose logs [service-name]`
- 💬 Check Discord/Slack channel


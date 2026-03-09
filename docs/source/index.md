# Simmons Portioning Tool - Complete Documentation

Welcome to the Simmons Portioning Tool documentation! This is your central hub for understanding, deploying, and extending the entire microservices stack.

## 📚 Documentation Sections

### 🚀 [Getting Started](getting-started/quick-start.md)
New to the project? Start here to get up and running quickly:
- **[Quick Start Guide](getting-started/quick-start.md)** - 5-minute setup
- **[Installation](getting-started/installation.md)** - Detailed setup instructions
- **[Running the Stack](getting-started/running-the-stack.md)** - Starting all services

### 🏗️ [Architecture](architecture/overview.md)
Understand how the system is designed:
- **[System Overview](architecture/overview.md)** - High-level architecture
- **[Microservices](architecture/microservices.md)** - Service breakdown
- **[Database Design](architecture/database-design.md)** - Data model and schema
- **[API Gateway](architecture/api-gateway.md)** - Unified entry point

### ⚙️ [Features](features/enumeration/overview.md)
Explore the core functionality:
- **Enumeration System** - SKU management and search
- **Mix Management** - Portioning mix configurations
- **Global Configuration** - System-wide settings

### 📡 [API Documentation](api/enumeration/overview.md)
Complete API reference with interactive documentation:
- **[Enumeration API](api/enumeration/overview.md)** - SKU and Mix endpoints
  - [Interactive Docs](http://localhost:8080/api/enumeration/docs) (when running)
- **[Global Config API](api/config/overview.md)** - Configuration endpoints
  - [Interactive Docs](http://localhost:8080/api/config/docs) (when running)

### 💾 [Database Management](database/quick-start.md)
Work with MongoDB:
- **[Quick Start](database/quick-start.md)** - Common database tasks
- **[Reinitialization](database/reinitialization.md)** - Reset and reinit database
- **[Schema Overview](database/schema.md)** - Collections and documents
- **[Indexes](database/indexes.md)** - Query optimization

### 🔧 [Operations](operations/docker-setup.md)
Deploy and manage the system:
- **[Docker Setup](operations/docker-setup.md)** - Container configuration
- **[Database Management](operations/database-management.md)** - MongoDB operations
- **[Monitoring](operations/monitoring.md)** - Health checks and metrics
- **[Troubleshooting](operations/troubleshooting.md)** - Common issues

### 👨‍💻 [Development](development/adding-microservices.md)
Extend and customize:
- **[Adding New Microservices](development/adding-microservices.md)** - Service onboarding
- **[Testing](development/testing.md)** - Unit and integration tests
- **[Contributing](development/contributing.md)** - Development guidelines

### 📋 [Specifications](specifications/enumeration-services.md)
Technical specifications and design documents:
- **[Enumeration Services Spec](specifications/enumeration-services.md)** - Service design
- **[System Design](specifications/system-design.md)** - Architecture patterns
- **[Technical Requirements](specifications/requirements.md)** - Functional requirements

---

## 🎯 Quick Navigation by Role

### **For New Developers**
1. [Quick Start Guide](getting-started/quick-start.md)
2. [System Architecture](architecture/overview.md)
3. [API Documentation](api/enumeration/overview.md)
4. [Running the Stack](getting-started/running-the-stack.md)

### **For DevOps/System Admins**
1. [Docker Setup](operations/docker-setup.md)
2. [Database Management](database/quick-start.md)
3. [Monitoring](operations/monitoring.md)
4. [Troubleshooting](operations/troubleshooting.md)

### **For Backend Developers**
1. [Architecture Overview](architecture/overview.md)
2. [Microservices Design](architecture/microservices.md)
3. [Adding New Services](development/adding-microservices.md)
4. [API Documentation](api/enumeration/overview.md)

### **For Data/ML Engineers**
1. [Database Schema](database/schema.md)
2. [Feature Documentation](features/enumeration/overview.md)
3. [MIX Configuration](features/mix/configuration.md)

---

## 🌐 Live API Documentation

When the stack is running locally:

| Service | Documentation | Base URL |
|---------|--------------|----------|
| **Enumeration API** | [Swagger UI](http://localhost:8080/api/enumeration/docs) | `http://localhost:8080/api/enumeration` |
| **Global Config API** | [Swagger UI](http://localhost:8080/api/config/docs) | `http://localhost:8080/api/config` |
| **API Gateway** | Health Check | `http://localhost:8080/health` |

---

## 📦 Project Structure

```
simmons-portioning-tool/
├── docs/                          # Documentation (you are here)
│   ├── mkdocs.yml                # MkDocs configuration
│   └── source/                   # Documentation source files
├── enumeration-api/              # Enumeration microservice
├── global-config-api/            # Configuration microservice
├── gateway/                       # Nginx API gateway
├── mongodb-init/                 # Database initialization
├── scripts/                       # Utility scripts
└── docker-compose.yml            # Service orchestration
```

---

## 🔗 Key Documentation Files

- **[Microservice API Onboarding](development/adding-microservices.md)** - How to add new services
- **[MIX Configuration Guide](features/mix/configuration.md)** - Configure plants, bird sizes, mfg types
- **[MongoDB Reinitialization](database/reinitialization.md)** - Reset database to clean state
- **[Database Quick Start](database/quick-start.md)** - Common MongoDB tasks

---

## 📞 Support & Contribution

- **Issues?** Check [Troubleshooting](operations/troubleshooting.md)
- **Want to contribute?** See [Contributing Guide](development/contributing.md)
- **Have questions?** Check relevant section or contact the team

---

**Last Updated:** March 2026


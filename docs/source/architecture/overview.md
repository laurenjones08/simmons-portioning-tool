# System Architecture Overview

The Simmons Portioning Tool is built as a modern microservices system with a unified API gateway, multiple specialized services, and centralized configuration management.

## High-Level Architecture

```
┌─────────────┐      ┌──────────────┐      ┌──────────────┐
│   Browser   │      │  API Client  │      │   Frontend   │
│  Docs Page  │      │   (Mobile)   │      │  (Streamlit) │
└──────┬──────┘      └──────┬───────┘      └──────┬───────┘
       │                     │                     │
       │                     │                     │
       ├─────────────────────┼─────────────────────┤
       │                     │                     │
       v                     v                     v
   ┌────────────────────────────────────────────────────┐
   │         API Gateway (Nginx on Port 8080)           │
   │  Unified entry point for all microservices         │
   │  ├─ Route /api/enumeration/* → Enumeration API    │
   │  ├─ Route /api/config/* → Global Config API       │
   │  └─ Route /docs → Documentation                    │
   └────────────────────────────────────────────────────┘
       │                     │
       │                     │
   ┌───v──────────────┐  ┌──v──────────────┐
   │ Enumeration API  │  │  Config API     │
   │ Port 8000        │  │  Port 8001      │
   │ ─────────────────┤  │ ─────────────── │
   │ • SKU Management │  │ • Config Key    │
   │ • MIX Config     │  │   Management    │
   │ • Search        │  │ • System Config │
   └───┬──────────────┘  └──┬──────────────┘
       │                     │
       │        ┌────────────┤
       │        │            │
       └────────┤            │
                v            v
        ┌──────────────────────────┐
        │   MongoDB (Port 27017)   │
        │ ──────────────────────── │
        │ • enumeration_db         │
        │   - skus collection      │
        │   - mixes collection     │
        │ • config_db              │
        │   - global_config        │
        └──────────────────────────┘
```

## Architecture Layers

### 1. Presentation Layer
- **Documentation Portal** - MkDocs based docs site (Port 3000)
- **Interactive API Docs** - Swagger UI for each service
- **Frontend** - Streamlit application for user interface

### 2. API Gateway Layer
- **Nginx** - Reverse proxy routing requests to services
- **Single Entry Point** - All clients connect to one URL
- **Health Monitoring** - Gateway health checks

### 3. Service Layer
Two independent microservices with identical architecture:

#### Service Architecture Pattern
```
HTTP Request
    │
    v
┌─────────────────────────────────┐
│   Router Layer (FastAPI)        │
│   • HTTP endpoints              │
│   • Request validation          │
│   • Response serialization      │
└──────────────┬──────────────────┘
               │
               v
┌─────────────────────────────────┐
│   Service Layer                 │
│   • Business logic              │
│   • Data orchestration          │
│   • Error handling              │
└──────────────┬──────────────────┘
               │
               v
┌─────────────────────────────────┐
│   Repository Layer              │
│   • Database queries            │
│   • Data mapping                │
│   • Query optimization          │
└──────────────┬──────────────────┘
               │
               v
┌─────────────────────────────────┐
│   MongoDB                       │
│   • Document storage            │
│   • Indexing                    │
│   • Transactions (if needed)    │
└─────────────────────────────────┘
```

### 4. Data Layer
- **MongoDB** - Document database for all services
- **Persistence** - Docker volumes for data persistence
- **Backup** - Database initialization scripts

## Microservices

### Enumeration API
- **Purpose**: Manage SKU (Stock Keeping Unit) data and MIX configurations
- **Port**: 8000 (internal)
- **Public Route**: `/api/enumeration/*`
- **Key Features**:
  - SKU CRUD operations
  - MIX management with uniqueness enforcement
  - Advanced search and filtering
  - Distributed tracing integration

### Global Config API
- **Purpose**: Centralized configuration management
- **Port**: 8001 (internal)
- **Public Route**: `/api/config/*`
- **Key Features**:
  - System-wide configuration storage
  - Type validation (int, string, float, bool)
  - Configuration value ranges
  - Audit trail with timestamps

## Data Model

### Two Independent Databases

```
MongoDB Instance
├── enumeration_db
│   ├── skus (Collection)
│   │   └── Indexes: tradeNumber, customerType, productType, etc.
│   └── mixes (Collection)
│       └── Indexes: mfgType, skuSetKey, includesFDS, etc.
│
└── config_db
    └── global_config (Collection)
        └── Configuration key-value pairs with metadata
```

## Communication Patterns

### Service-to-Service Communication
Services communicate with each other via HTTP through the gateway:
- No direct service-to-service calls
- All communication goes through the gateway
- Enables monitoring and logging at the gateway level

### Client Communication
```
Client
  │
  v
http://api-gateway:8080/api/[service]/[endpoint]
  │
  v
Gateway (Nginx)
  │
  ├─ /api/enumeration/* → Enumeration API:8000
  ├─ /api/config/* → Global Config API:8001
  └─ /docs → Documentation
```

## Networking

### Docker Network
All services communicate via `mongo_net` bridge network:
- Service names resolve to internal IPs
- No need for hardcoded IPs or hostnames
- Isolated from host network by default

### Port Mapping
```
External Host      Container Network     Service
─────────────────────────────────────────────────
localhost:8080   → Gateway:80          Nginx
localhost:27017  → MongoDB:27017       MongoDB
localhost:16686  → Jaeger:16686        Jaeger (tracing)
localhost:3000   → Docs:8000           MkDocs (documentation)
```

## Deployment Architecture

### Containers and Orchestration
```
docker-compose.yml orchestrates:
├── mongodb (mongo:7.0)
├── enumeration-api (Python 3.11 + FastAPI)
├── global-config-api (Python 3.11 + FastAPI)
├── api-gateway (Nginx 1.27)
├── jaeger (jaegertracing/all-in-one)
└── mkdocs (squidfunk/mkdocs-material)

Volume Management:
├── mongo_data (persistent MongoDB storage)
└── Code mounts (hot reload for development)
```

## Key Design Principles

### 1. Separation of Concerns
- Each service owns its domain
- Clear boundaries between services
- Independent deployability

### 2. Layered Architecture
- Router → Service → Repository pattern
- Dependency injection for testability
- Clear responsibilities at each layer

### 3. Scalability
- Stateless services (can be scaled horizontally)
- Database as single source of truth
- Cacheable responses where applicable

### 4. Observability
- Distributed tracing via Jaeger
- Structured logging from all services
- Health check endpoints

### 5. Maintainability
- Type hints throughout (Python)
- Comprehensive documentation
- Clear naming conventions

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **API Framework** | FastAPI | 0.109.0 |
| **Web Server** | Uvicorn | 0.27.0 |
| **Database** | MongoDB | 7.0 |
| **API Gateway** | Nginx | 1.27 |
| **Validation** | Pydantic | 2.5.3 |
| **Driver** | PyMongo | 4.6.1 |
| **Tracing** | Jaeger | 1.53 |
| **Documentation** | MkDocs | Latest |
| **Container Runtime** | Docker | Latest |
| **Orchestration** | Docker Compose | Latest |

## Data Flow Example: Create a MIX

```
1. Client Request
   POST /api/enumeration/mixes
   Body: { skus: {...}, mfgType: "DSI", ... }

2. Gateway (Nginx)
   Routes to enumeration-api:8000

3. Enumeration API Router
   ├─ Validates request against MixCreate schema
   ├─ Injects MixService
   └─ Calls service.create_mix()

4. MixService
   ├─ Creates MIX instance
   ├─ Computes skuSetKey for uniqueness
   ├─ Calls repository.create()
   └─ Returns MIX or raises ValueError on conflict

5. MixRepository
   ├─ Builds MongoDB document
   ├─ Checks unique index (mfgType + skuSetKey)
   └─ Inserts or raises DuplicateKeyError

6. Response to Client
   201 Created + MIX document (JSON)
   or
   409 Conflict (if mix already exists)
```

## Next Steps

- [Microservices Design](microservices.md) - Deep dive into each service
- [Database Design](database-design.md) - Schema and indexing details
- [API Gateway](api-gateway.md) - Routing and gateway configuration


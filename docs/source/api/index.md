# API Documentation

Browse the complete API reference for Simmons Portioning Tool.

## Available APIs

### Enumeration API
Manage SKU (Stock Keeping Unit) data and MIX configurations.

**Base URL:** `http://localhost:8080/api/enumeration`

**Documentation:**
- Swagger UI: `http://localhost:8080/api/enumeration/docs`
- ReDoc: `http://localhost:8080/api/enumeration/redoc`

**Key Endpoints:**
- `GET /health` - Health check
- `POST /mixes` - Create mix
- `GET /mixes/{mix_id}` - Get specific mix
- `POST /mixes/search` - Search mixes
- `PUT /mixes/{mix_id}` - Update mix
- `DELETE /mixes/{mix_id}` - Delete mix

### Global Config API
Manage system configuration and settings.

**Base URL:** `http://localhost:8080/api/config`

**Documentation:**
- Swagger UI: `http://localhost:8080/api/config/docs`
- ReDoc: `http://localhost:8080/api/config/redoc`

**Key Endpoints:**
- `GET /health` - Health check
- `GET /{key}` - Get configuration value
- `GET /` - List all configuration
- `PUT /{key}` - Update configuration value

## Interactive Documentation

When the stack is running, visit the interactive API documentation to:
- ✅ Test endpoints directly
- ✅ View request/response examples
- ✅ Explore parameters and schemas
- ✅ Download OpenAPI specifications

See detailed documentation:
- [Enumeration API](enumeration/overview.md)
- [Global Config API](config/overview.md)


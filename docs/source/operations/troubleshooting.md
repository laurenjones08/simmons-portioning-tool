# Troubleshooting Guide

Solutions to common issues and problems.

## API Issues

### 502 Bad Gateway

**Symptom:** Getting 502 error from API gateway.

**Causes & Solutions:**
1. API service not started - `docker compose logs enumeration-api`
2. API crashed - check logs for errors
3. Port conflict - ensure ports 8000, 8001 are free

**Fix:**
```bash
docker compose restart enumeration-api global-config-api
```

### 404 Not Found

**Symptom:** Endpoint returns 404 even though it should exist.

**Causes & Solutions:**
1. Wrong route path - check [API documentation](../api/enumeration/overview.md)
2. Service not in gateway config - verify `/gateway/nginx.conf`
3. Typo in request - double-check URL

**Verify routes:**
```bash
curl http://localhost:8080/  # See available routes
```

### Connection Refused

**Symptom:** Cannot connect to http://localhost:8080

**Causes & Solutions:**
1. Gateway not running - `docker compose ps api-gateway`
2. Port 8080 in use - change in docker-compose.yml
3. Docker daemon not running - restart Docker

**Fix:**
```bash
docker compose up -d api-gateway
```

## Database Issues

### Cannot Connect to MongoDB

**Symptom:** "MongoServerSelectionError: connect ECONNREFUSED"

**Causes & Solutions:**
1. MongoDB not running - `docker compose up -d mongodb`
2. Wrong credentials - check MONGODB_URL in compose
3. MongoDB still initializing - wait 5-10 seconds

**Test connection:**
```bash
docker exec mongodb mongosh -u root -p example --eval "db.runCommand({ping: 1})"
```

### Database Not Initialized

**Symptom:** Collections don't exist when querying.

**Solution:** Run reinitialization:
```powershell
.\scripts\reinit-mongodb.ps1
```

### Unique Index Violation

**Symptom:** Cannot create MIX - "duplicate key error"

**Cause:** MIX already exists with same SKU set + mfgType.

**Solution:**
- Use different mfgType, OR
- Delete existing MIX first, OR
- Modify the SKU set

### Data Loss After Restart

**Symptom:** Data disappeared when restarting containers.

**Cause:** Using `docker compose down -v`

**Prevention:**
```bash
# DO NOT use -v flag unless you want to delete data
docker compose down      # Safe
docker compose down -v   # Deletes all data!
```

## Performance Issues

### Queries are Slow

**Causes & Solutions:**
1. Missing indexes - check [Indexes](../database/indexes.md)
2. Too much data - limit query results
3. Complex search - use simpler filters

**Check index usage:**
```bash
docker exec mongodb mongosh -u root -p example --eval "
  db.getSiblingDB('enumeration_db').mixes.aggregate([{
    \$indexStats: {}
  }]).pretty()
"
```

### High Memory Usage

**Causes & Solutions:**
1. Large data export - paginate results
2. Memory leak in service - restart service
3. Docker resource limits - increase in docker settings

**Check resource usage:**
```bash
docker stats
```

## Container Issues

### Container Won't Start

**Symptom:** Container exits immediately or is "unhealthy".

**Debug:**
```bash
docker compose logs enumeration-api
```

**Common causes:**
- Port already in use
- Volume permission issue
- Configuration error

**Fix:**
```bash
docker compose build --no-cache enumeration-api
docker compose up -d --force-recreate enumeration-api
```

### Container Keeps Restarting

**Symptom:** Container starts then stops repeatedly.

**Debug:**
```bash
docker compose logs --tail=50 [service-name]
```

**Fix:** Identify error in logs and fix configuration or code.

### Out of Disk Space

**Symptom:** Docker commands fail with "no space left"

**Solutions:**
```bash
# See what's using space
docker system df

# Remove unused data
docker system prune -a

# Remove specific old images
docker image rm [image-id]
```

## Network Issues

### Services Can't Communicate

**Symptom:** "Cannot reach enumeration-api from gateway"

**Causes & Solutions:**
1. Services not on same network - check docker-compose.yml
2. Firewall blocking - check Docker desktop settings
3. Service name DNS resolution - restart Docker

**Test network:**
```bash
docker exec api-gateway ping enumeration-api
```

### Port Already in Use

**Symptom:** "Bind for 0.0.0.0:8080 failed: port is already allocated"

**Solution:** Change port in docker-compose.yml:
```yaml
api-gateway:
  ports:
    - "8888:80"  # Changed from 8080
```

Then restart:
```bash
docker compose down
docker compose up -d
```

## Documentation Issues

### Docs not Loading

**Symptom:** Cannot access http://localhost:3000

**Causes & Solutions:**
1. Docs container not running - `docker compose ps docs`
2. Port 3000 in use - change port in compose
3. Browser cache - clear cache or use incognito

**Restart docs:**
```bash
docker compose restart docs
```

## Getting Help

1. **Check logs:** `docker compose logs [service-name]`
2. **Review this guide** - most issues listed above
3. **Check API docs** - functionality reference at `http://localhost:8080/api/*/docs`
4. **Rebuild clean:** `docker compose down -v && docker compose up -d --build`



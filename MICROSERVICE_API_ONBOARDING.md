# Unified API Gateway: Adding New Microservices

This project exposes APIs through a single gateway so frontend clients can use one base URL.

- Public entrypoint: `http://localhost:8080`
- Existing routes:
  - `http://localhost:8080/api/enumeration/*` -> `enumeration-api:8000/*`
  - `http://localhost:8080/api/config/*` -> `global-config-api:8001/*`

## Why this pattern

Using one gateway now makes local development closer to AWS API Gateway behavior later:
- one host for frontend
- path-based routing per service
- easier CORS/auth rollout
- fewer frontend environment variables

## Add a new microservice (checklist)

- [ ] Create service source folder and Dockerfile (for example `reporting-api/`)
- [ ] Add the service to `docker-compose.yml`
- [ ] Add a gateway route in `gateway/nginx.conf`
- [ ] Add docs endpoint/health checks for the new service
- [ ] Verify through gateway (not direct service port)

## 1) Add service to Docker Compose

Add a service block in `docker-compose.yml`:

```yaml
  reporting-api:
    build:
      context: ./reporting-api
      dockerfile: Dockerfile
    container_name: reporting-api
    restart: always
    environment:
      MONGODB_URL: mongodb://root:example@mongodb:27017
      MONGODB_DATABASE: reporting_db
      SERVICE_NAME: reporting-api
    expose:
      - "8002"
    volumes:
      - ./reporting-api:/app
    networks:
      - mongo_net
    depends_on:
      mongodb:
        condition: service_healthy
      jaeger:
        condition: service_started
```

Notes:
- Use `expose` for internal connectivity.
- Keep the frontend talking to the gateway on `:8080`.

## 2) Add routing rule in Nginx

Edit `gateway/nginx.conf` and add:

```nginx
    # /api/reporting/* -> reporting-api:8002/*
    location /api/reporting/ {
        proxy_pass http://reporting-api:8002/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
```

## 3) Rebuild and start

```powershell
docker compose up -d --build
```

## 4) Validate route

```powershell
curl http://localhost:8080/health
curl http://localhost:8080/api/enumeration/health
curl http://localhost:8080/api/config/health
curl http://localhost:8080/api/reporting/health
```

## Route design conventions

Follow this URL pattern for consistency:
- `http://localhost:8080/api/<service-name>/<endpoint>`

Examples:
- `.../api/enumeration/skus/search`
- `.../api/config/health`
- `.../api/reporting/reports/daily`

## Preparing for AWS API Gateway

When migrating to AWS API Gateway:
- keep the same path prefixes (`/api/enumeration`, `/api/config`, ...)
- map each prefix to its backend integration
- preserve frontend base URL semantics by environment

This keeps frontend code changes minimal.

## Troubleshooting

### 404 from gateway
- Confirm route exists in `gateway/nginx.conf`
- Confirm path prefix includes trailing slash in location block
- Restart gateway: `docker compose restart api-gateway`

### 502 from gateway
- Backend service is down or wrong target port
- Check service logs: `docker compose logs reporting-api`
- Confirm container is on `mongo_net`

### Works on direct port, fails through gateway
- Check proxy path mapping (`proxy_pass` trailing slash matters)
- Verify service expects root-path requests


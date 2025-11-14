# Docker

## Overview

Docker packages applications into containers - lightweight, isolated environments with all dependencies. Build once, run anywhere.

## Core Concepts

### Images vs Containers
- **Image**: Blueprint (read-only template)
- **Container**: Running instance of image

```bash
# Build image
docker build -t myapp:1.0 .

# Run container from image
docker run myapp:1.0
```

### Dockerfile

```dockerfile
# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 5000

# Run command
CMD ["python", "app.py"]
```

### Multi-stage Builds

Multi-stage builds reduce image size by separating build and runtime environments:

```dockerfile
# Stage 1: Build
FROM node:18 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 2: Production
FROM node:18-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY package*.json ./
USER node
EXPOSE 3000
CMD ["node", "dist/index.js"]
```

**Benefits**:
- Smaller final image (only runtime dependencies)
- Build tools not in production image
- Better security (fewer attack surfaces)

**Advanced Example** (Go application):

```dockerfile
# Build stage
FROM golang:1.21-alpine AS builder
WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o app .

# Final stage
FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /build/app .
EXPOSE 8080
CMD ["./app"]
```

**Using Build Arguments**:

```dockerfile
FROM node:${NODE_VERSION:-18}-alpine AS base
ARG BUILD_ENV=production

FROM base AS builder
WORKDIR /app
COPY . .
RUN npm ci --only=${BUILD_ENV}
RUN npm run build

FROM base AS production
COPY --from=builder /app/dist ./dist
CMD ["node", "dist/server.js"]
```

```bash
# Build with custom arguments
docker build --build-arg NODE_VERSION=20 --build-arg BUILD_ENV=development -t myapp:dev .
```

## Docker Commands

```bash
# Build
docker build -t myapp:1.0 .

# Run
docker run -p 8000:5000 myapp:1.0
docker run -d -p 8000:5000 myapp:1.0  # Detached

# View containers
docker ps          # Running
docker ps -a       # All

# View images
docker images

# Logs
docker logs container_id

# Stop container
docker stop container_id

# Remove
docker rm container_id
docker rmi image_name

# Execute command in running container
docker exec -it container_id /bin/bash
docker exec container_id ls /app

# Copy files to/from container
docker cp ./file.txt container_id:/app/
docker cp container_id:/app/logs.txt ./

# Inspect container details
docker inspect container_id
docker inspect --format='{{.NetworkSettings.IPAddress}}' container_id

# View container resource usage
docker stats
docker stats container_id

# Create image from container
docker commit container_id myimage:tag

# Save/load images
docker save myimage:tag > image.tar
docker load < image.tar

# Export/import containers
docker export container_id > container.tar
docker import container.tar myimage:tag

# Prune unused resources
docker system prune       # Remove unused data
docker system prune -a    # Remove all unused images
docker volume prune       # Remove unused volumes
docker network prune      # Remove unused networks
```

## Docker Compose

Multiple containers together:

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:5000"
    environment:
      DATABASE_URL: postgres://db:5432/mydb
    depends_on:
      - db

  db:
    image: postgres:13
    environment:
      POSTGRES_PASSWORD: secret
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

```bash
docker-compose up         # Start all services
docker-compose up -d      # Start in detached mode
docker-compose down       # Stop all services
docker-compose logs -f    # Follow logs
docker-compose ps         # List containers
docker-compose exec web bash  # Execute command in service
docker-compose build      # Build or rebuild services
docker-compose restart    # Restart services
docker-compose scale web=3    # Scale service (deprecated, use --scale)
docker-compose up --scale web=3  # Scale service
```

### Advanced Docker Compose

**Environment Files**:

```yaml
# docker-compose.yml
services:
  web:
    build: .
    env_file:
      - .env
      - .env.production
    environment:
      - NODE_ENV=production
      - API_KEY=${API_KEY}  # From .env file
```

**Profiles** (selective service startup):

```yaml
services:
  web:
    image: nginx
    profiles: ["frontend"]

  api:
    image: node:18
    # Always starts (no profile)

  debug:
    image: busybox
    profiles: ["debug"]
```

```bash
# Start only services without profiles
docker-compose up

# Start with specific profile
docker-compose --profile frontend up
docker-compose --profile debug up
```

**Override Files** (environment-specific configs):

```yaml
# docker-compose.override.yml (auto-loaded in development)
services:
  web:
    volumes:
      - ./src:/app/src  # Hot reload in dev
    environment:
      - DEBUG=true
```

```yaml
# docker-compose.prod.yml
services:
  web:
    restart: always
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
```

```bash
# Use specific override
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
```

**Depends On with Health Checks**:

```yaml
services:
  db:
    image: postgres:15
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  web:
    build: .
    depends_on:
      db:
        condition: service_healthy  # Wait for healthy status
```

**Networks in Compose**:

```yaml
services:
  frontend:
    networks:
      - frontend-net

  backend:
    networks:
      - frontend-net
      - backend-net

  database:
    networks:
      - backend-net  # Isolated from frontend

networks:
  frontend-net:
    driver: bridge
  backend-net:
    driver: bridge
    internal: true  # No external access
```

## Docker Networking

### Network Types

**Bridge Network** (default, isolated):
```bash
# Create custom bridge network
docker network create my-network

# Run containers on same network
docker run -d --name db --network my-network postgres
docker run -d --name app --network my-network myapp

# Containers can communicate via container names
# app can connect to: postgres://db:5432
```

**Host Network** (share host's network stack):
```bash
# No port mapping needed, uses host ports directly
docker run --network host nginx
# nginx now accessible on host's port 80
```

**None Network** (no networking):
```bash
docker run --network none myapp
# Completely isolated, no network access
```

**Overlay Network** (multi-host, Docker Swarm):
```bash
docker network create --driver overlay my-overlay
# Enables container communication across multiple Docker hosts
```

### Network Operations

```bash
# List networks
docker network ls

# Inspect network
docker network inspect my-network

# Connect running container to network
docker network connect my-network container_id

# Disconnect from network
docker network disconnect my-network container_id

# Remove network
docker network rm my-network
```

### Service Discovery

Containers on same network can resolve each other by name:

```bash
# Start containers
docker network create app-net
docker run -d --name redis --network app-net redis
docker run -d --name web --network app-net myapp

# Inside 'web' container:
# ping redis  # Works!
# curl http://redis:6379  # Works!
```

### Network Aliases

```bash
docker run -d --network my-net --network-alias db1 --network-alias database postgres
# Accessible as 'db1' or 'database' from other containers
```

### Port Publishing Modes

```bash
# Publish to specific host port
docker run -p 8080:80 nginx

# Publish to random host port
docker run -p 80 nginx

# Publish all exposed ports to random ports
docker run -P nginx

# Publish to specific interface
docker run -p 127.0.0.1:8080:80 nginx

# UDP ports
docker run -p 53:53/udp dns-server
```

## Resource Management

### CPU and Memory Limits

**Container Resource Constraints**:

```bash
# Limit memory
docker run -m 512m nginx  # Max 512MB RAM
docker run --memory=1g --memory-reservation=750m myapp

# Limit CPU
docker run --cpus=1.5 myapp  # Max 1.5 CPU cores
docker run --cpu-shares=512 myapp  # Relative weight (default 1024)

# Combine limits
docker run -m 1g --cpus=2 --name myapp myimage
```

**Docker Compose Resource Limits**:

```yaml
services:
  web:
    image: nginx
    deploy:
      resources:
        limits:
          cpus: '0.50'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
```

### Storage Limits

```bash
# Limit container disk usage
docker run --storage-opt size=10G myapp

# Set read/write limits (bytes per second)
docker run --device-read-bps /dev/sda:1mb myapp
docker run --device-write-bps /dev/sda:1mb myapp
```

### Process and File Descriptor Limits

```bash
# Limit number of processes
docker run --pids-limit 100 myapp

# Limit file descriptors
docker run --ulimit nofile=1024:2048 myapp

# Multiple ulimits
docker run \
  --ulimit nofile=1024:2048 \
  --ulimit nproc=512:1024 \
  myapp
```

### Restart Policies

```bash
# No restart (default)
docker run --restart=no myapp

# Always restart
docker run --restart=always myapp

# Restart on failure only
docker run --restart=on-failure:5 myapp  # Max 5 retries

# Restart unless explicitly stopped
docker run --restart=unless-stopped myapp
```

**In Docker Compose**:

```yaml
services:
  web:
    image: nginx
    restart: unless-stopped

  worker:
    image: myworker
    restart: on-failure
```

### Health Checks

**Dockerfile**:

```dockerfile
FROM nginx
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost/ || exit 1
```

**Docker Run**:

```bash
docker run \
  --health-cmd="curl -f http://localhost/ || exit 1" \
  --health-interval=30s \
  --health-timeout=3s \
  --health-retries=3 \
  nginx
```

**Docker Compose**:

```yaml
services:
  web:
    image: nginx
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 40s
```

**Check Health Status**:

```bash
docker ps  # Shows (healthy), (unhealthy), (health: starting)
docker inspect --format='{{.State.Health.Status}}' container_id
```

## Security Patterns

### Running as Non-Root User

**Dockerfile**:

```dockerfile
FROM node:18-alpine

# Create app user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

WORKDIR /app
COPY --chown=nodejs:nodejs . .

# Switch to non-root user
USER nodejs

EXPOSE 3000
CMD ["node", "server.js"]
```

**At Runtime**:

```bash
docker run --user 1001:1001 myapp
docker run --user nobody myapp
```

### Secrets Management

**Docker Secrets** (Swarm mode):

```bash
# Create secret
echo "my-secret-password" | docker secret create db_password -

# Use in service
docker service create \
  --secret db_password \
  --name myapp \
  myimage

# Inside container, secret available at:
# /run/secrets/db_password
```

**Docker Compose with Secrets**:

```yaml
version: '3.8'

services:
  db:
    image: postgres
    secrets:
      - db_password
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password

secrets:
  db_password:
    file: ./secrets/db_password.txt
```

**Build Secrets** (BuildKit):

```dockerfile
# syntax=docker/dockerfile:1
FROM alpine

# Mount secret during build only (not in final image)
RUN --mount=type=secret,id=github_token \
    TOKEN=$(cat /run/secrets/github_token) && \
    git clone https://$TOKEN@github.com/private/repo.git
```

```bash
docker build --secret id=github_token,src=./token.txt -t myapp .
```

### Read-only Root Filesystem

```bash
docker run --read-only --tmpfs /tmp myapp
```

```yaml
services:
  web:
    image: nginx
    read_only: true
    tmpfs:
      - /tmp
      - /var/run
```

### Security Options

```bash
# Drop all capabilities, add only needed ones
docker run \
  --cap-drop=ALL \
  --cap-add=NET_BIND_SERVICE \
  nginx

# Disable privilege escalation
docker run --security-opt=no-new-privileges:true myapp

# AppArmor profile
docker run --security-opt apparmor=docker-default myapp

# Seccomp profile
docker run --security-opt seccomp=profile.json myapp
```

### Image Scanning

```bash
# Scan image for vulnerabilities (requires Docker Scout)
docker scout cve myimage:latest

# Using Trivy (third-party)
docker run aquasec/trivy image myimage:latest

# Scan during build in CI/CD
docker build -t myapp .
docker scout cve myapp
```

### Best Practices Summary

**Security Checklist**:
- ✓ Use official base images
- ✓ Run as non-root user
- ✓ Use specific image tags (not :latest)
- ✓ Scan images for vulnerabilities
- ✓ Use secrets management (never hardcode)
- ✓ Minimize attack surface (multi-stage builds)
- ✓ Keep images updated
- ✓ Use read-only filesystems where possible
- ✓ Limit container capabilities

## Image Optimization

### Layer Caching Strategy

Order Dockerfile commands by change frequency (least to most):

```dockerfile
# Anti-pattern: Cache invalidated on any code change
FROM node:18
WORKDIR /app
COPY . .                    # ❌ Copies everything first
RUN npm install             # ❌ Reinstalls on any file change

# Best practice: Maximize cache reuse
FROM node:18
WORKDIR /app
COPY package*.json ./       # ✓ Only dependencies
RUN npm ci                  # ✓ Cached unless package.json changes
COPY . .                    # ✓ Code copied last
RUN npm run build           # ✓ Only rebuilds if code changed
```

### Minimize Image Size

**Use Minimal Base Images**:

```dockerfile
# Large: 1.1GB
FROM ubuntu:22.04

# Medium: 350MB
FROM node:18

# Small: 180MB
FROM node:18-slim

# Smallest: 120MB
FROM node:18-alpine
```

**Remove Build Dependencies**:

```dockerfile
# Install and cleanup in single layer
FROM alpine:3.18
RUN apk add --no-cache --virtual .build-deps \
        gcc \
        musl-dev \
        postgresql-dev \
    && pip install --no-cache-dir psycopg2 \
    && apk del .build-deps  # Remove build tools
```

**Combine Commands**:

```dockerfile
# Bad: Creates 3 layers
RUN apt-get update
RUN apt-get install -y curl
RUN rm -rf /var/lib/apt/lists/*

# Good: Single layer
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*
```

**Use .dockerignore**:

```dockerignore
# .dockerignore
node_modules
npm-debug.log
.git
.env
*.md
.vscode
dist
coverage
.pytest_cache
__pycache__
```

### BuildKit Optimizations

Enable BuildKit for better performance:

```bash
export DOCKER_BUILDKIT=1
docker build .
```

**Cache Mounts** (don't include in final image):

```dockerfile
# syntax=docker/dockerfile:1
FROM node:18

WORKDIR /app
COPY package*.json ./

# Mount npm cache, faster rebuilds
RUN --mount=type=cache,target=/root/.npm \
    npm ci

COPY . .
RUN npm run build
```

**Parallel Builds**:

```dockerfile
# syntax=docker/dockerfile:1
FROM alpine AS fetch-deps
RUN apk add --no-cache curl
RUN curl -O https://example.com/file1.tar.gz

FROM alpine AS build
COPY --from=fetch-deps /file1.tar.gz .
RUN tar -xzf file1.tar.gz && make

FROM alpine
COPY --from=build /app .
```

### Analyzing Image Size

```bash
# View image layers and sizes
docker history myimage:latest

# Show layer details
docker history --no-trunc myimage:latest

# Use dive for interactive analysis
docker run --rm -it \
    -v /var/run/docker.sock:/var/run/docker.sock \
    wagoodman/dive:latest myimage:latest
```

## Debugging & Troubleshooting

### Debugging Containers

**Access Running Container**:

```bash
# Interactive shell
docker exec -it container_id /bin/sh
docker exec -it container_id /bin/bash

# Run specific command
docker exec container_id ps aux
docker exec container_id cat /var/log/app.log
```

**Debug Non-Starting Container**:

```bash
# Override entrypoint to investigate
docker run -it --entrypoint /bin/sh myimage

# Check why container exited
docker logs container_id
docker inspect container_id --format='{{.State.ExitCode}}'
```

**Copy Files for Analysis**:

```bash
# Copy logs out
docker cp container_id:/var/log/app.log ./app.log

# Copy config file in
docker cp ./new-config.yml container_id:/etc/config.yml
```

### Build Debugging

**Show Build Output**:

```bash
# No cache, show all output
docker build --no-cache --progress=plain .

# Stop at specific stage for debugging
docker build --target builder -t debug-image .
docker run -it debug-image /bin/sh
```

**Check Build Context**:

```bash
# See what's being sent to Docker daemon
docker build --no-cache . 2>&1 | grep "Sending build context"
```

### Common Issues

**Issue: Container Exits Immediately**

```bash
# Check logs
docker logs container_id

# Common causes:
# 1. Main process exits (use tail -f, or proper daemon)
# 2. Command not found
# 3. Permission issues

# Debug:
docker run -it myimage /bin/sh  # Override CMD
```

**Issue: Cannot Connect to Container**

```bash
# Check if port is published
docker port container_id

# Check if service is listening
docker exec container_id netstat -tlnp

# Check network
docker inspect container_id --format='{{.NetworkSettings.IPAddress}}'
```

**Issue: Out of Disk Space**

```bash
# Check Docker disk usage
docker system df

# Detailed view
docker system df -v

# Clean up
docker system prune -a  # Remove all unused
docker volume prune     # Remove unused volumes
docker image prune -a   # Remove unused images
```

**Issue: DNS Resolution Fails**

```bash
# Test DNS in container
docker exec container_id nslookup google.com
docker exec container_id cat /etc/resolv.conf

# Set custom DNS
docker run --dns 8.8.8.8 --dns 8.8.4.4 myimage
```

### Monitoring and Logs

**Real-time Logs**:

```bash
# Follow logs
docker logs -f container_id

# Last 100 lines
docker logs --tail 100 container_id

# Since specific time
docker logs --since 2024-01-01T00:00:00 container_id

# Multiple containers
docker-compose logs -f web db
```

**Resource Monitoring**:

```bash
# Real-time stats
docker stats

# Single container
docker stats container_id

# No streaming (snapshot)
docker stats --no-stream

# Custom format
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

**Events**:

```bash
# Watch Docker events
docker events

# Filter events
docker events --filter type=container
docker events --filter event=start
```

### Performance Troubleshooting

**Slow Container**:

```bash
# Check resource usage
docker stats container_id

# Check if CPU/memory limited
docker inspect container_id --format='{{.HostConfig.Memory}}'
docker inspect container_id --format='{{.HostConfig.CpuShares}}'

# Check I/O
docker exec container_id iostat
```

**Slow Build**:

```bash
# Enable BuildKit for better performance
export DOCKER_BUILDKIT=1

# Use build cache from registry
docker build --cache-from myregistry/myapp:latest .

# Parallel builds
docker build --parallel .
```

## Best Practices

1. **Small images**: Use minimal base images (alpine)
2. **Layer caching**: Order commands by change frequency
3. **Security**: Don't run as root, use secrets
4. **Health checks**: Monitor container health
5. **Resource limits**: Always set memory/CPU limits in production
6. **Logging**: Use structured logging, log to stdout/stderr
7. **Secrets**: Never hardcode, use Docker secrets or env vars
8. **Single process**: One process per container
9. **Immutable infrastructure**: Rebuild images, don't modify running containers

```dockerfile
# Good: Optimized production image
FROM python:3.9-slim

# Non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Dependencies first (caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY --chown=appuser:appuser . .

# Switch to non-root
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

EXPOSE 8000
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
```

## Volumes

### Volume Types

**Named Volumes** (managed by Docker):
```bash
# Create named volume
docker volume create my-data

# Use volume
docker run -v my-data:/app/data myapp

# Inspect volume
docker volume inspect my-data

# List volumes
docker volume ls

# Remove volume
docker volume rm my-data
```

**Bind Mounts** (host filesystem):
```bash
# Mount host directory (absolute path required)
docker run -v /host/path:/container/path myapp

# Read-only mount
docker run -v /host/path:/container/path:ro myapp

# Use with relative path (requires pwd)
docker run -v "$(pwd)":/app myapp
```

**tmpfs Mounts** (in-memory, temporary):
```bash
# Temporary in-memory storage
docker run --tmpfs /tmp myapp

# With size limit
docker run --tmpfs /tmp:rw,size=100m myapp
```

### Volume Operations

```bash
# Create with driver options
docker volume create --driver local \
    --opt type=nfs \
    --opt o=addr=192.168.1.1,rw \
    --opt device=:/path/to/dir \
    my-nfs-volume

# Copy data between volumes
docker run --rm \
    -v old-volume:/from \
    -v new-volume:/to \
    alpine sh -c "cp -av /from/* /to/"

# Backup volume
docker run --rm \
    -v my-volume:/data \
    -v $(pwd):/backup \
    alpine tar czf /backup/backup.tar.gz -C /data .

# Restore volume
docker run --rm \
    -v my-volume:/data \
    -v $(pwd):/backup \
    alpine sh -c "rm -rf /data/* && tar xzf /backup/backup.tar.gz -C /data"
```

### Volumes in Docker Compose

```yaml
version: '3.8'

services:
  db:
    image: postgres:15
    volumes:
      # Named volume
      - postgres-data:/var/lib/postgresql/data
      # Bind mount
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql:ro
      # Anonymous volume
      - /var/lib/postgresql

  app:
    build: .
    volumes:
      # Development: hot reload
      - ./src:/app/src
      # Don't overwrite node_modules
      - /app/node_modules

volumes:
  postgres-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/postgres
```

### Volume Backup Strategies

**Automated Backup Script**:

```bash
#!/bin/bash
VOLUME_NAME="postgres-data"
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

docker run --rm \
    -v $VOLUME_NAME:/source:ro \
    -v $BACKUP_DIR:/backup \
    alpine tar czf /backup/${VOLUME_NAME}_${DATE}.tar.gz -C /source .

# Keep only last 7 backups
find $BACKUP_DIR -name "${VOLUME_NAME}_*.tar.gz" -mtime +7 -delete
```

**Database-Specific Backups**:

```bash
# PostgreSQL dump
docker exec postgres_container pg_dump -U user dbname > backup.sql

# MySQL dump
docker exec mysql_container mysqldump -u user -p dbname > backup.sql

# MongoDB dump
docker exec mongo_container mongodump --out /backup
```

## CI/CD Integration

### Building Images in Pipelines

**GitHub Actions**:

```yaml
name: Docker Build and Push

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: |
            myorg/myapp:latest
            myorg/myapp:${{ github.sha }}
          cache-from: type=registry,ref=myorg/myapp:latest
          cache-to: type=inline
```

**GitLab CI**:

```yaml
# .gitlab-ci.yml
stages:
  - build
  - test
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker tag $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA $CI_REGISTRY_IMAGE:latest
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    - docker push $CI_REGISTRY_IMAGE:latest
```

**Jenkins Pipeline**:

```groovy
pipeline {
    agent any

    environment {
        DOCKER_IMAGE = "myorg/myapp"
        DOCKER_TAG = "${env.BUILD_NUMBER}"
    }

    stages {
        stage('Build') {
            steps {
                script {
                    docker.build("${DOCKER_IMAGE}:${DOCKER_TAG}")
                }
            }
        }

        stage('Test') {
            steps {
                script {
                    docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}").inside {
                        sh 'npm test'
                    }
                }
            }
        }

        stage('Push') {
            steps {
                script {
                    docker.withRegistry('https://registry.hub.docker.com', 'docker-credentials') {
                        docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}").push()
                        docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}").push('latest')
                    }
                }
            }
        }
    }
}
```

### Registry Operations

**Docker Hub**:

```bash
# Login
docker login

# Tag image
docker tag myapp:latest myusername/myapp:1.0

# Push to Docker Hub
docker push myusername/myapp:1.0

# Pull from Docker Hub
docker pull myusername/myapp:1.0
```

**Private Registry**:

```bash
# Run private registry
docker run -d -p 5000:5000 --name registry registry:2

# Tag for private registry
docker tag myapp localhost:5000/myapp:1.0

# Push to private registry
docker push localhost:5000/myapp:1.0

# Pull from private registry
docker pull localhost:5000/myapp:1.0
```

**Amazon ECR**:

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    123456789.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag myapp:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/myapp:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/myapp:latest
```

**Google Container Registry (GCR)**:

```bash
# Configure Docker for GCR
gcloud auth configure-docker

# Tag and push
docker tag myapp gcr.io/project-id/myapp:latest
docker push gcr.io/project-id/myapp:latest
```

### Multi-platform Builds

```bash
# Create builder
docker buildx create --name multiplatform --use

# Build for multiple platforms
docker buildx build \
    --platform linux/amd64,linux/arm64,linux/arm/v7 \
    -t myorg/myapp:latest \
    --push .
```

### Container Deployment Patterns

**Blue-Green Deployment**:

```bash
# Deploy new version (green)
docker run -d --name app-green -p 8081:8080 myapp:v2

# Test green deployment
curl http://localhost:8081/health

# Switch traffic (update load balancer or swap ports)
docker stop app-blue
docker rm app-blue
docker run -d --name app-blue -p 8080:8080 myapp:v2
docker stop app-green
docker rm app-green
```

**Rolling Update with Docker Swarm**:

```bash
# Initialize swarm
docker swarm init

# Create service
docker service create \
    --name myapp \
    --replicas 3 \
    --update-parallelism 1 \
    --update-delay 10s \
    myapp:v1

# Update service (rolling update)
docker service update --image myapp:v2 myapp
```

## Development Workflows

### Hot Reload Development

**Node.js with Nodemon**:

```dockerfile
# Dockerfile.dev
FROM node:18
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
CMD ["npm", "run", "dev"]  # Uses nodemon
```

```yaml
# docker-compose.dev.yml
services:
  web:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - ./src:/app/src  # Hot reload on changes
      - /app/node_modules  # Don't overwrite
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=development
```

**Python with Flask**:

```dockerfile
# Dockerfile.dev
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["flask", "run", "--host=0.0.0.0", "--reload"]
```

```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - ./app:/app  # Hot reload
    environment:
      - FLASK_ENV=development
      - FLASK_DEBUG=1
```

### Debugging in Containers

**Node.js Debugging**:

```yaml
services:
  web:
    build: .
    ports:
      - "9229:9229"  # Debug port
    command: node --inspect=0.0.0.0:9229 server.js
    volumes:
      - ./src:/app/src
```

**Python Debugging (pdb)**:

```yaml
services:
  api:
    build: .
    stdin_open: true  # Enable interactive mode
    tty: true
    command: python -m pdb app.py
```

**Remote Debugging with VS Code**:

```json
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Docker: Attach to Node",
      "type": "node",
      "request": "attach",
      "port": 9229,
      "address": "localhost",
      "localRoot": "${workspaceFolder}",
      "remoteRoot": "/app",
      "protocol": "inspector"
    }
  ]
}
```

### Local Development Environment

**Complete Dev Stack**:

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Frontend
  frontend:
    build: ./frontend
    volumes:
      - ./frontend/src:/app/src
      - /app/node_modules
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:4000

  # Backend API
  backend:
    build: ./backend
    volumes:
      - ./backend/src:/app/src
      - /app/node_modules
    ports:
      - "4000:4000"
      - "9229:9229"  # Debug port
    environment:
      - NODE_ENV=development
      - DATABASE_URL=postgres://user:pass@db:5432/mydb
      - REDIS_URL=redis://redis:6379
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started

  # PostgreSQL
  db:
    image: postgres:15-alpine
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=mydb
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 5s
      timeout: 5s
      retries: 5

  # Redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  # Nginx (reverse proxy)
  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx/dev.conf:/etc/nginx/nginx.conf:ro
    ports:
      - "80:80"
    depends_on:
      - frontend
      - backend

volumes:
  postgres-data:
  redis-data:
```

### Testing Workflows

**Run Tests in Container**:

```bash
# Run tests
docker-compose run --rm backend npm test

# Run specific test
docker-compose run --rm backend npm test -- user.test.js

# Run with coverage
docker-compose run --rm backend npm run test:coverage
```

**Integration Tests**:

```yaml
# docker-compose.test.yml
services:
  test:
    build: .
    command: npm test
    environment:
      - NODE_ENV=test
      - DATABASE_URL=postgres://test:test@test-db:5432/testdb
    depends_on:
      - test-db

  test-db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=test
      - POSTGRES_PASSWORD=test
      - POSTGRES_DB=testdb
    tmpfs:
      - /var/lib/postgresql/data  # In-memory for speed
```

```bash
# Run integration tests
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

## Real-World Example: Full-Stack Application

### Project Structure

```
myapp/
├── frontend/
│   ├── Dockerfile
│   ├── Dockerfile.dev
│   └── src/
├── backend/
│   ├── Dockerfile
│   ├── Dockerfile.dev
│   └── src/
├── nginx/
│   ├── nginx.conf
│   └── ssl/
├── docker-compose.yml
├── docker-compose.prod.yml
└── docker-compose.dev.yml
```

### Frontend Dockerfile (React)

```dockerfile
# Build stage
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine
COPY --from=builder /app/build /usr/share/nginx/html
COPY nginx/default.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Backend Dockerfile (Node.js API)

```dockerfile
# Dependencies stage
FROM node:18-alpine AS deps
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

# Build stage
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Production stage
FROM node:18-alpine
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001
WORKDIR /app
COPY --from=deps --chown=nodejs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nodejs:nodejs /app/dist ./dist
COPY --chown=nodejs:nodejs package*.json ./
USER nodejs
EXPOSE 4000
CMD ["node", "dist/index.js"]
```

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  frontend:
    image: myorg/frontend:${VERSION:-latest}
    restart: unless-stopped
    networks:
      - frontend-net

  backend:
    image: myorg/backend:${VERSION:-latest}
    restart: unless-stopped
    environment:
      - NODE_ENV=production
      - DATABASE_URL_FILE=/run/secrets/db_url
      - JWT_SECRET_FILE=/run/secrets/jwt_secret
    secrets:
      - db_url
      - jwt_secret
    depends_on:
      - db
      - redis
    networks:
      - frontend-net
      - backend-net
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '0.5'
          memory: 512M

  db:
    image: postgres:15-alpine
    restart: unless-stopped
    environment:
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
    secrets:
      - db_password
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - backend-net
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 1G

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes:
      - redis-data:/data
    networks:
      - backend-net

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/prod.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - nginx-cache:/var/cache/nginx
    depends_on:
      - frontend
      - backend
    networks:
      - frontend-net

networks:
  frontend-net:
  backend-net:
    internal: true

volumes:
  postgres-data:
  redis-data:
  nginx-cache:

secrets:
  db_url:
    file: ./secrets/db_url.txt
  db_password:
    file: ./secrets/db_password.txt
  jwt_secret:
    file: ./secrets/jwt_secret.txt
```

### Deployment Commands

```bash
# Build images
docker-compose -f docker-compose.prod.yml build

# Tag for registry
docker tag myapp_frontend:latest myorg/frontend:1.0.0
docker tag myapp_backend:latest myorg/backend:1.0.0

# Push to registry
docker push myorg/frontend:1.0.0
docker push myorg/backend:1.0.0

# Deploy to production
VERSION=1.0.0 docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale backend=5

# View logs
docker-compose -f docker-compose.prod.yml logs -f backend

# Rollback
VERSION=0.9.0 docker-compose -f docker-compose.prod.yml up -d
```

## ELI10

Docker is like shipping containers for code:
- Package everything needed (dependencies, code, config)
- Send it anywhere (laptop, server, cloud)
- Runs the same everywhere!

No more "it works on my machine" problems!

## Further Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Hub Images](https://hub.docker.com/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

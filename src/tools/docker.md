# Docker

Docker is a platform for developing, shipping, and running applications in containers. Containers package software with all dependencies, ensuring consistent behavior across different environments.

## Overview

Docker enables developers to package applications with their dependencies into standardized units called containers, which can run anywhere Docker is installed.

**Key Concepts:**
- **Container**: Lightweight, standalone executable package
- **Image**: Read-only template for creating containers
- **Dockerfile**: Script defining how to build an image
- **Registry**: Repository for storing and distributing images
- **Docker Hub**: Public registry for Docker images

## Installation

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Verify installation
docker --version
docker run hello-world
```

## Basic Commands

### Container Operations

```bash
# Run a container
docker run nginx
docker run -d nginx  # Detached mode
docker run -it ubuntu bash  # Interactive terminal

# Run with options
docker run -d \
  --name my-nginx \
  -p 8080:80 \
  -v /host/path:/container/path \
  -e ENV_VAR=value \
  nginx

# List containers
docker ps  # Running containers
docker ps -a  # All containers

# Stop/Start containers
docker stop container_name
docker start container_name
docker restart container_name

# Remove containers
docker rm container_name
docker rm -f container_name  # Force remove
docker container prune  # Remove all stopped containers
```

### Image Operations

```bash
# List images
docker images
docker image ls

# Pull image from registry
docker pull nginx
docker pull nginx:1.21

# Build image from Dockerfile
docker build -t myapp:1.0 .
docker build -t myapp:latest -f Dockerfile.prod .

# Remove images
docker rmi image_name
docker image prune  # Remove unused images
docker image prune -a  # Remove all unused images

# Tag image
docker tag myapp:1.0 username/myapp:1.0

# Push to registry
docker push username/myapp:1.0
```

### Logs and Debugging

```bash
# View logs
docker logs container_name
docker logs -f container_name  # Follow logs
docker logs --tail 100 container_name

# Execute command in container
docker exec container_name ls /app
docker exec -it container_name bash

# Inspect container
docker inspect container_name
docker stats container_name  # Resource usage

# Copy files
docker cp file.txt container_name:/path/
docker cp container_name:/path/file.txt ./
```

## Dockerfile

### Basic Dockerfile

```dockerfile
# Base image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy application code
COPY . .

# Expose port
EXPOSE 3000

# Set environment variables
ENV NODE_ENV=production

# Run command
CMD ["node", "server.js"]
```

### Multi-stage Build

```dockerfile
# Build stage
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Production stage
FROM node:18-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY package*.json ./
RUN npm install --production
EXPOSE 3000
CMD ["node", "dist/server.js"]
```

### Dockerfile Instructions

```dockerfile
# FROM: Base image
FROM ubuntu:22.04

# LABEL: Metadata
LABEL maintainer="dev@example.com"
LABEL version="1.0"

# ENV: Environment variables
ENV APP_HOME=/app
ENV PORT=8080

# ARG: Build-time variables
ARG VERSION=latest
RUN echo "Building version ${VERSION}"

# WORKDIR: Set working directory
WORKDIR /app

# COPY: Copy files from host
COPY src/ /app/src/

# ADD: Copy and extract archives
ADD archive.tar.gz /app/

# RUN: Execute commands during build
RUN apt-get update && \
    apt-get install -y python3 && \
    rm -rf /var/lib/apt/lists/*

# USER: Set user
USER appuser

# EXPOSE: Document ports
EXPOSE 8080 8443

# VOLUME: Create mount point
VOLUME ["/data"]

# ENTRYPOINT: Configure container executable
ENTRYPOINT ["python3"]

# CMD: Default arguments for ENTRYPOINT
CMD ["app.py"]

# HEALTHCHECK: Container health check
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost/ || exit 1
```

## Docker Compose

### Basic docker-compose.yml

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8080:80"
    volumes:
      - ./src:/app/src
    environment:
      - NODE_ENV=development
    depends_on:
      - db

  db:
    image: postgres:15
    volumes:
      - db-data:/var/lib/postgresql/data
    environment:
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: myapp

volumes:
  db-data:
```

### Docker Compose Commands

```bash
# Start services
docker-compose up
docker-compose up -d  # Detached

# Stop services
docker-compose down
docker-compose down -v  # Remove volumes

# Build services
docker-compose build
docker-compose build --no-cache

# View logs
docker-compose logs
docker-compose logs -f service_name

# Execute commands
docker-compose exec web bash
docker-compose exec db psql -U postgres

# Scale services
docker-compose up -d --scale web=3

# List services
docker-compose ps
```

### Advanced Compose Configuration

```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.dev
      args:
        VERSION: "1.0"
    image: myapp:latest
    container_name: myapp
    restart: unless-stopped
    ports:
      - "3000:3000"
    volumes:
      - ./src:/app/src:ro  # Read-only
      - node_modules:/app/node_modules
    environment:
      NODE_ENV: development
      DATABASE_URL: postgres://db:5432/myapp
    env_file:
      - .env
    depends_on:
      db:
        condition: service_healthy
    networks:
      - backend
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:15-alpine
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    environment:
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    networks:
      - backend

networks:
  backend:
    driver: bridge

volumes:
  postgres_data:
  node_modules:
```

## Networking

### Network Commands

```bash
# List networks
docker network ls

# Create network
docker network create mynetwork
docker network create --driver bridge mynetwork

# Connect container to network
docker network connect mynetwork container_name

# Disconnect from network
docker network disconnect mynetwork container_name

# Inspect network
docker network inspect mynetwork

# Remove network
docker network rm mynetwork
```

### Network Types

```bash
# Bridge (default)
docker run --network bridge nginx

# Host (use host's network)
docker run --network host nginx

# None (no networking)
docker run --network none nginx

# Custom bridge network
docker network create app-network
docker run --network app-network --name web nginx
docker run --network app-network --name db postgres
```

## Volumes

### Volume Management

```bash
# Create volume
docker volume create myvolume

# List volumes
docker volume ls

# Inspect volume
docker volume inspect myvolume

# Remove volume
docker volume rm myvolume
docker volume prune  # Remove unused volumes

# Use volume in container
docker run -v myvolume:/data nginx
docker run --mount source=myvolume,target=/data nginx
```

### Volume Types

```bash
# Named volume
docker run -v myvolume:/app/data nginx

# Bind mount (host directory)
docker run -v /host/path:/container/path nginx
docker run -v $(pwd):/app nginx

# Anonymous volume
docker run -v /container/path nginx

# Read-only volume
docker run -v myvolume:/data:ro nginx
```

## Best Practices

### Dockerfile Optimization

```dockerfile
# 1. Use specific image tags
FROM node:18.16-alpine  # Good
FROM node:latest        # Avoid

# 2. Minimize layers
RUN apt-get update && apt-get install -y \
    package1 \
    package2 \
    && rm -rf /var/lib/apt/lists/*

# 3. Order instructions by frequency of change
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./     # Changes less frequently
RUN npm install
COPY . .                  # Changes more frequently

# 4. Use .dockerignore
# Create .dockerignore file:
# node_modules
# .git
# .env
# *.log

# 5. Don't run as root
RUN addgroup -g 1001 appgroup && \
    adduser -D -u 1001 -G appgroup appuser
USER appuser

# 6. Use multi-stage builds
FROM node:18 AS builder
WORKDIR /app
COPY . .
RUN npm run build

FROM node:18-alpine
COPY --from=builder /app/dist /app/dist
```

### Security Best Practices

```bash
# 1. Scan images for vulnerabilities
docker scan myimage:latest

# 2. Use official images
docker pull nginx:alpine

# 3. Keep images updated
docker pull nginx:latest

# 4. Limit container resources
docker run --memory="512m" --cpus="1.0" nginx

# 5. Run as non-root user
docker run --user 1000:1000 nginx

# 6. Use secrets for sensitive data
docker secret create db_password password.txt
docker service create --secret db_password myapp
```

## Common Patterns

### Development Environment

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  app:
    build:
      context: .
      target: development
    volumes:
      - .:/app
      - /app/node_modules
    ports:
      - "3000:3000"
    environment:
      NODE_ENV: development
    command: npm run dev
```

### Production Setup

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  app:
    image: myapp:${VERSION:-latest}
    restart: always
    ports:
      - "80:3000"
    environment:
      NODE_ENV: production
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1'
          memory: 512M
```

### Backup Script

```bash
#!/bin/bash
# Backup Docker volume

VOLUME_NAME="mydata"
BACKUP_FILE="backup-$(date +%Y%m%d-%H%M%S).tar.gz"

docker run --rm \
  -v ${VOLUME_NAME}:/data \
  -v $(pwd):/backup \
  alpine \
  tar czf /backup/${BACKUP_FILE} -C /data .

echo "Backup created: ${BACKUP_FILE}"
```

## Troubleshooting

### Common Issues

```bash
# Container exits immediately
docker logs container_name
docker run -it container_name sh

# Port already in use
docker ps -a | grep 8080
lsof -i :8080

# Out of disk space
docker system df
docker system prune  # Remove unused data
docker system prune -a  # Remove all unused data

# Permission denied
sudo usermod -aG docker $USER
newgrp docker

# Network issues
docker network ls
docker network inspect bridge

# Image pull errors
docker pull --platform linux/amd64 image_name
```

### Debugging Commands

```bash
# Inspect container
docker inspect --format='{{.State.Status}}' container_name
docker inspect --format='{{.NetworkSettings.IPAddress}}' container_name

# Container events
docker events --filter container=container_name

# System information
docker info
docker version

# Resource usage
docker stats
docker top container_name
```

## Useful Aliases

```bash
# Add to ~/.bashrc or ~/.zshrc
alias dps='docker ps'
alias dpsa='docker ps -a'
alias di='docker images'
alias drm='docker rm'
alias drmi='docker rmi'
alias dstop='docker stop $(docker ps -q)'
alias dclean='docker system prune -af'
alias dlog='docker logs -f'
alias dexec='docker exec -it'
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `docker run` | Create and start container |
| `docker ps` | List running containers |
| `docker stop` | Stop container |
| `docker rm` | Remove container |
| `docker images` | List images |
| `docker pull` | Download image |
| `docker build` | Build image from Dockerfile |
| `docker push` | Upload image to registry |
| `docker logs` | View container logs |
| `docker exec` | Run command in container |
| `docker-compose up` | Start services |
| `docker-compose down` | Stop services |

Docker simplifies application deployment and ensures consistency across development, testing, and production environments.

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
docker-compose down       # Stop all services
docker-compose logs -f    # Follow logs
```

## Best Practices

1. **Small images**: Use minimal base images (alpine)
2. **Layer caching**: Order commands by change frequency
3. **Security**: Don't run as root, use secrets
4. **Health checks**: Monitor container health

```dockerfile
# Good: Minimal image
FROM python:3.9-slim
RUN pip install --no-cache-dir -r requirements.txt

# Health check
HEALTHCHECK --interval=30s CMD curl -f http://localhost/health
```

## Volumes

```bash
# Mount host directory
docker run -v /host/path:/container/path myapp

# Named volume
docker run -v myvolume:/data myapp

# View volumes
docker volume ls
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

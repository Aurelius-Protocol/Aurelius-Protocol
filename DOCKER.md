# Aurelius Validator Docker Guide

This guide covers Docker deployment of the Aurelius Protocol validator for the Bittensor subnet (testnet 290).

## Quick Start

```bash
# Pull the latest image
docker pull ghcr.io/aurelius-protocol/aurelius-validator:latest

# Or from Docker Hub
docker pull aureliusprotocol/validator:latest

# Run with minimal configuration
docker run -d \
  --name aurelius-validator \
  -p 8091:8091 \
  --env-file .env \
  -v ~/.bittensor/wallets:/home/aurelius/.bittensor/wallets:ro \
  -v aurelius-data:/var/lib/aurelius \
  ghcr.io/aurelius-protocol/aurelius-validator:latest
```

## Prerequisites

- Docker 20.10+ or Docker Desktop
- Docker Compose v2.0+ (optional, for compose deployments)
- A registered Bittensor wallet on subnet 290
- API keys (OpenAI required, Chutes optional)

## Image Registries

The validator image is published to two registries:

| Registry | Image |
|----------|-------|
| GitHub Container Registry | `ghcr.io/aurelius-protocol/aurelius-validator` |
| Docker Hub | `aureliusprotocol/validator` |

## Configuration

### Required Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Copy the example configuration
cp .env.example .env

# Edit with your settings
nano .env
```

**Minimum required variables:**

```bash
# Chat provider: "chutes" (recommended) or "openai"
CHAT_PROVIDER=chutes
CHUTES_API_KEY=your-chutes-api-key

# OpenAI - ALWAYS required for moderation API
OPENAI_API_KEY=your-openai-api-key

# Bittensor network
BT_NETWORK=test
BT_NETUID=290

# Wallet configuration
VALIDATOR_WALLET_NAME=validator
VALIDATOR_HOTKEY=default
```

### Wallet Setup

Mount your Bittensor wallet directory as read-only:

```bash
-v ~/.bittensor/wallets:/home/aurelius/.bittensor/wallets:ro
```

The wallet structure should be:
```
~/.bittensor/wallets/
└── <VALIDATOR_WALLET_NAME>/
    ├── coldkey
    ├── coldkeypub.txt
    └── hotkeys/
        └── <VALIDATOR_HOTKEY>
```

## Deployment Methods

### Method 1: Docker Run

Basic deployment:

```bash
docker run -d \
  --name aurelius-validator \
  --restart unless-stopped \
  -p 8091:8091 \
  --env-file .env \
  -v ~/.bittensor/wallets:/home/aurelius/.bittensor/wallets:ro \
  -v aurelius-data:/var/lib/aurelius \
  ghcr.io/aurelius-protocol/aurelius-validator:latest
```

Production deployment with resource limits:

```bash
docker run -d \
  --name aurelius-validator \
  --restart always \
  -p 8091:8091 \
  --env-file .env \
  --security-opt no-new-privileges:true \
  --memory 4g \
  --cpus 2 \
  -v ~/.bittensor/wallets:/home/aurelius/.bittensor/wallets:ro \
  -v /var/lib/aurelius:/var/lib/aurelius \
  ghcr.io/aurelius-protocol/aurelius-validator:latest
```

### Method 2: Docker Compose (Recommended)

**Development:**

```bash
# Clone repository and navigate to it
cd Aurelius-Protocol

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start validator
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

**Production:**

```bash
# Create data directories first
sudo mkdir -p /var/lib/aurelius/datasets
sudo chown -R 1000:1000 /var/lib/aurelius

# Start with production overrides
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Method 3: Build Locally

```bash
# Build the image
docker build -t aurelius-validator:local .

# Run with local image
docker run -d \
  --name aurelius-validator \
  -p 8091:8091 \
  --env-file .env \
  -v ~/.bittensor/wallets:/home/aurelius/.bittensor/wallets:ro \
  -v aurelius-data:/var/lib/aurelius \
  aurelius-validator:local
```

## Volume Mounts

| Container Path | Purpose | Required | Type |
|---------------|---------|----------|------|
| `/home/aurelius/.bittensor/wallets` | Bittensor wallet files | Yes | Bind (read-only) |
| `/var/lib/aurelius/datasets` | JSONL dataset logs | Recommended | Named or bind |
| `/var/lib/aurelius/miner_scores.json` | Miner scoring data | Recommended | Named or bind |
| `/var/lib/aurelius/validator_trust.json` | Validator trust data | Recommended | Named or bind |

### Using Named Volumes (Development)

Named volumes are managed by Docker and persist across container restarts:

```yaml
volumes:
  - aurelius-data:/var/lib/aurelius
```

List volumes: `docker volume ls`
Inspect: `docker volume inspect aurelius-data`
Remove: `docker volume rm aurelius-data`

### Using Bind Mounts (Production)

Bind mounts map host directories directly, making backups easier:

```yaml
volumes:
  - /var/lib/aurelius:/var/lib/aurelius
```

## Environment Variables

The container sets these defaults (can be overridden):

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_DATASET_PATH` | `/var/lib/aurelius/datasets` | Dataset storage path |
| `MINER_SCORES_PATH` | `/var/lib/aurelius/miner_scores.json` | Scores file path |
| `VALIDATOR_TRUST_PERSISTENCE_PATH` | `/var/lib/aurelius/validator_trust.json` | Trust data path |
| `VALIDATOR_HOST` | `0.0.0.0` | Listen address |
| `BT_PORT_VALIDATOR` | `8091` | Validator port |
| `AUTO_DETECT_EXTERNAL_IP` | `true` | Auto-detect public IP |

See `.env.example` for all available configuration options.

## Multi-Architecture Support

Images are built for both AMD64 and ARM64 architectures:

```bash
# AMD64 (Intel/AMD x86_64)
docker pull --platform linux/amd64 ghcr.io/aurelius-protocol/aurelius-validator:latest

# ARM64 (Apple Silicon M1/M2/M3, AWS Graviton, Raspberry Pi 4)
docker pull --platform linux/arm64 ghcr.io/aurelius-protocol/aurelius-validator:latest
```

## Available Tags

| Tag | Description |
|-----|-------------|
| `latest` | Latest stable release from main branch |
| `v1.0.0` | Specific version |
| `v1.0` | Latest patch for minor version |
| `v1` | Latest minor for major version |
| `sha-abc1234` | Specific commit |
| `main` | Latest from main branch (same as latest) |

## Health Checks

The container includes a health check that verifies port 8091 is accepting connections.

Check health status:

```bash
# Single container
docker inspect --format='{{.State.Health.Status}}' aurelius-validator

# With compose
docker compose ps
```

Health states: `starting`, `healthy`, `unhealthy`

## Backup and Restore

### Backup Data

```bash
# Stop container first (optional but recommended)
docker stop aurelius-validator

# Backup named volumes
docker run --rm \
  -v aurelius-data:/data:ro \
  -v $(pwd):/backup \
  alpine tar czf /backup/aurelius-backup-$(date +%Y%m%d).tar.gz -C /data .

# Restart
docker start aurelius-validator
```

### Restore Data

```bash
# Stop container
docker stop aurelius-validator

# Restore to volume
docker run --rm \
  -v aurelius-data:/data \
  -v $(pwd):/backup:ro \
  alpine tar xzf /backup/aurelius-backup-20240115.tar.gz -C /data

# Start container
docker start aurelius-validator
```

## Logs

### View Logs

```bash
# Docker run
docker logs aurelius-validator
docker logs -f aurelius-validator  # Follow
docker logs --tail 100 aurelius-validator  # Last 100 lines

# Docker Compose
docker compose logs
docker compose logs -f validator
docker compose logs --tail 100
```

### Log Rotation

Logs are automatically rotated (configured in compose):
- Max size: 100MB per file
- Max files: 5

## Troubleshooting

### Shell Access

```bash
docker exec -it aurelius-validator /bin/bash
```

### Check Running Processes

```bash
docker exec aurelius-validator ps aux
```

### Verify Network Connectivity

```bash
# Check if port is listening inside container
docker exec aurelius-validator nc -z localhost 8091 && echo "Port open" || echo "Port closed"

# Check from host
nc -z localhost 8091 && echo "Port accessible" || echo "Port not accessible"
```

### Common Issues

**Wallet not found:**
```
Error: Wallet not found
```
- Verify wallet path: `ls -la ~/.bittensor/wallets/`
- Check mount: `docker exec aurelius-validator ls -la /home/aurelius/.bittensor/wallets/`
- Ensure wallet names match environment variables

**Permission denied on wallet:**
```
PermissionError: [Errno 13] Permission denied
```
- Check file permissions: `ls -la ~/.bittensor/wallets/*/hotkeys/`
- Wallet files should be readable by UID 1000
- Try: `chmod 644 ~/.bittensor/wallets/*/hotkeys/*`

**Port already in use:**
```
Bind for 0.0.0.0:8091 failed: port is already allocated
```
- Check what's using the port: `lsof -i :8091`
- Use different host port: `-p 8092:8091`

**Container keeps restarting:**
```bash
# Check logs for errors
docker logs aurelius-validator

# Check exit code
docker inspect aurelius-validator --format='{{.State.ExitCode}}'
```

**Out of memory:**
- Increase memory limit in docker-compose.yml or docker run
- Check memory usage: `docker stats aurelius-validator`

## Security Best Practices

1. **Never include secrets in the image** - Use environment variables or mounted files
2. **Use read-only wallet mounts** - Prevents accidental modification: `:ro`
3. **Run as non-root** - Container runs as `aurelius` user (UID 1000)
4. **Keep images updated** - Pull latest for security patches
5. **Use `no-new-privileges`** - Prevents privilege escalation in production
6. **Limit resources** - Set memory and CPU limits to prevent runaway processes

## Building for Development

```bash
# Build with build arguments
docker build \
  --build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
  --build-arg VCS_REF=$(git rev-parse --short HEAD) \
  --build-arg VERSION=$(git describe --tags --always) \
  -t aurelius-validator:dev .

# Build for specific platform
docker buildx build --platform linux/amd64 -t aurelius-validator:amd64 .
docker buildx build --platform linux/arm64 -t aurelius-validator:arm64 .

# Build multi-arch and load locally
docker buildx build --platform linux/amd64,linux/arm64 -t aurelius-validator:multi --push .
```

## CI/CD Integration

The GitHub Actions workflow (`.github/workflows/docker-publish.yml`) automatically:

1. Builds on push to `main` and version tags
2. Creates multi-architecture images (AMD64 + ARM64)
3. Pushes to both Docker Hub and GHCR
4. Runs security scans with Trivy
5. Updates Docker Hub description from README

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `DOCKERHUB_USERNAME` | Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token (not password) |

`GITHUB_TOKEN` is automatically provided by GitHub Actions.

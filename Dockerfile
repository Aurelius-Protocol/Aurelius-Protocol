# =============================================================================
# Aurelius Protocol Validator - Production Dockerfile
# =============================================================================
# Multi-stage build for minimal image size
# Supports AMD64 and ARM64 architectures
#
# Build:
#   docker build -t aurelius-validator .
#
# Run:
#   docker run -d --name validator \
#     -p 8091:8091 \
#     --env-file .env \
#     -v ~/.bittensor/wallets:/home/aurelius/.bittensor/wallets:ro \
#     -v aurelius-data:/var/lib/aurelius \
#     aurelius-validator
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies and compile wheels
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /build

# Copy dependency specification first (for better layer caching)
COPY pyproject.toml ./

# Copy package structure for editable install
COPY aurelius/ ./aurelius/
COPY validator.py ./

# Install the package with dependencies
RUN pip install --no-cache-dir -e .

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Minimal production image
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# Build arguments for labels
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# OCI Labels (https://github.com/opencontainers/image-spec/blob/main/annotations.md)
LABEL org.opencontainers.image.title="Aurelius Protocol Validator" \
      org.opencontainers.image.description="Bittensor subnet validator for AI alignment research (testnet 290)" \
      org.opencontainers.image.url="https://github.com/Aurelius-Protocol/Aurelius-Protocol" \
      org.opencontainers.image.source="https://github.com/Aurelius-Protocol/Aurelius-Protocol" \
      org.opencontainers.image.documentation="https://github.com/Aurelius-Protocol/Aurelius-Protocol/blob/main/DOCKER.md" \
      org.opencontainers.image.vendor="Aurelius Protocol" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.base.name="python:3.12-slim-bookworm" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${VERSION}"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    # For health checks
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 aurelius \
    && useradd --uid 1000 --gid aurelius --shell /bin/bash --create-home aurelius

# Create data directories with proper ownership
RUN mkdir -p /var/lib/aurelius/datasets \
    && mkdir -p /app \
    && mkdir -p /home/aurelius/.bittensor/wallets \
    && chown -R aurelius:aurelius /var/lib/aurelius \
    && chown -R aurelius:aurelius /app \
    && chown -R aurelius:aurelius /home/aurelius/.bittensor

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code (excluding miner via .dockerignore)
COPY --chown=aurelius:aurelius aurelius/validator/ ./aurelius/validator/
COPY --chown=aurelius:aurelius aurelius/shared/ ./aurelius/shared/
COPY --chown=aurelius:aurelius aurelius/__init__.py ./aurelius/
COPY --chown=aurelius:aurelius validator.py ./
COPY --chown=aurelius:aurelius pyproject.toml ./

# Entrypoint script to handle volume initialization
COPY --chown=aurelius:aurelius docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Switch to non-root user
USER aurelius

# Environment defaults for container deployment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Data paths inside container
    LOCAL_DATASET_PATH=/var/lib/aurelius/datasets \
    MINER_SCORES_PATH=/var/lib/aurelius/miner_scores.json \
    VALIDATOR_TRUST_PERSISTENCE_PATH=/var/lib/aurelius/validator_trust.json \
    # Network binding - listen on all interfaces
    VALIDATOR_HOST=0.0.0.0 \
    BT_PORT_VALIDATOR=8091 \
    # Auto-detect external IP for Bittensor registration
    AUTO_DETECT_EXTERNAL_IP=true

# Expose validator port
EXPOSE 8091

# Health check - verify port is accepting connections
# The bittensor axon may not respond to HTTP GET, so we use netcat
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD nc -z localhost 8091 || exit 1

# Entry point - initialize volumes then run validator
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command (can be overridden)
CMD ["python", "validator.py"]

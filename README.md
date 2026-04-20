# Aurelius v3

A Bittensor subnet for moral reasoning alignment data.

Miners submit structured ethical dilemma scenario configurations. Validators evaluate submissions using a quality classifier and run accepted scenarios through Concordia generative agent simulations. The simulation transcripts become training data that improves LLM performance on moral reasoning benchmarks (MoReBench).

## Architecture

```
Miner                    Validator                    Central API
  |                         |                             |
  |  ScenarioConfigSynapse  |                             |
  |<------------------------|                             |
  |  (config + work_id)     |                             |
  |------------------------>|                             |
  |                         |-- version check             |
  |                         |-- schema validate           |
  |                         |-- work-token balance ------>|
  |                         |-- rate limit check          |
  |                         |-- novelty check ----------->|
  |                         |-- classifier gate           |
  |                         |-- Concordia simulation      |
  |                         |-- work-token deduct ------->|
  |                         |-- set weights (on-chain)    |
  |                         |-- report submission ------->|
```

## Quick Start

```bash
# One-command setup for validators:
./scripts/quickstart.sh validator

# One-command setup for miners:
./scripts/quickstart.sh miner

# One-command setup for Central API:
./scripts/quickstart.sh api
```

Or step by step:

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install (choose components)
pip install -e "."              # Core (miner + validator)
pip install -e ".[api]"         # Central API
pip install -e ".[ml]"          # Classifier + embeddings
pip install -e ".[simulation]"  # Concordia runner
pip install -e ".[dev]"         # Development tools

# Generate secrets
./scripts/generate-env.sh

# Start infrastructure
docker compose up -d  # PostgreSQL + API

# Run validator
aurelius-validator

# Run miner
aurelius-miner
```

## Configuration

Two-tier system:

**Local** (environment variables / `.env`):
- `BT_SUBTENSOR_NETWORK` — `finney`, `test`, or `local`
- `BT_NETUID` — Subnet UID (37 mainnet, 290 testnet)
- `WALLET_NAME` / `WALLET_HOTKEY` — Bittensor wallet
- `CENTRAL_API_URL` — Central API endpoint
- `LLM_API_KEY` / `LLM_MODEL` / `LLM_BASE_URL` — For Concordia simulation (OpenAI-compatible, default: DeepSeek)

**Remote** (fetched from Central API `/config`):
- Polling interval, classifier threshold, novelty threshold
- Rate limits, work-token cost, Concordia settings
- Minimum miner/validator versions

See [`.env.example`](.env.example) for all local config options.

## Project Structure

```
aurelius/
  common/          Shared types, enums, schema, version, embeddings, classifier
  protocol.py      ScenarioConfigSynapse (Bittensor wire protocol)
  config.py        Two-tier configuration (local + remote)
  miner/           Miner Axon server, config store, work-token generation
  validator/       Validation pipeline, rate limiter, remote config, API client
  api/             Central API (FastAPI + PostgreSQL)
  simulation/      Concordia runner, translator, Docker isolation, transcripts
  benchmark/       Fine-tuning, MoReBench evaluation, influence scoring
  cli/             CLI tools (deposit verification)
  tools/           Seed dataset generation
```

## Testing

```bash
pip install -e ".[dev,ml]"

# Fast tests (no network, no Docker)
pytest tests/ --ignore=tests/api --ignore=tests/common/test_embeddings.py

# API tests (uses in-memory SQLite)
pytest tests/api/

# All tests including slow embedding tests
pytest tests/

# E2E tests (requires testnet)
pytest tests/e2e/ -m e2e
```

## License

MIT

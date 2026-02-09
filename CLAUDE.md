# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bittensor subnet (37 mainnet, 290 testnet) for AI alignment research. Validators receive prompts from miners, generate LLM responses, moderate content, check novelty, and score submissions. Scores become on-chain weights that determine miner rewards.

**Data flow**: Miners → Validators (port 8091) → Data Collector API → PostgreSQL with pgvector

## Commands

```bash
# Run all tests (uses uv, not bare pytest/python)
uv run pytest

# Run a single test file
uv run pytest tests/test_moral_experiment.py

# Run a single test function
uv run pytest tests/test_moral_experiment.py::test_function_name -v

# Lint and format
uv run ruff check .
uv run ruff check --fix .
uv run ruff format .

# Type check
uv run mypy aurelius

# Pre-commit (ruff + mypy + file cleanup)
pre-commit run --all-files

# Install in dev mode
uv pip install -e ".[dev]"
```

**Note**: `python` is not available in this environment — use `python3` or `uv run` for all commands.

## Architecture

### Entry Points

- `validator.py` / `miner.py` — CLI wrappers that call `aurelius.validator.validator:main` and `aurelius.miner.miner:main`

### Core Package (`aurelius/`)

```
aurelius/
├── validator/
│   ├── validator.py          # Main Validator class (~2400 lines)
│   └── experiments/          # Multi-experiment framework
│       ├── base.py           # Experiment, PushExperiment, PullExperiment ABCs
│       ├── manager.py        # ExperimentManager — routing, rate limits, merged weights
│       ├── prompt/           # Default "prompt" experiment (dangerous prompt discovery)
│       └── moral_reasoning/  # "moral-reasoning" experiment (moral dilemma evaluation)
├── miner/
│   ├── miner.py              # Miner logic (discover validators, send prompts)
│   └── registration.py       # Experiment registration CLI
└── shared/                   # Shared modules used by both validator and miner
    ├── protocol.py           # Synapse definitions (PromptSynapse, PullRequestSynapse, etc.)
    ├── config.py             # Network-aware config (auto-detects mainnet/testnet defaults)
    ├── chat_client.py        # LLM API with fallback (Chutes/DeepSeek → OpenAI)
    ├── moderation.py         # OpenAI moderation API
    ├── novelty_client.py     # Novelty detection via central API
    ├── experiment_client.py  # Sync experiment definitions from central API
    ├── scoring.py            # Miner scoring with windowed statistics
    ├── rate_limiter.py       # Per-miner, per-experiment rate limiting
    ├── consensus.py          # Multi-validator consensus verification
    ├── dataset_logger.py     # JSONL audit trail + central API submission
    └── telemetry/            # OpenTelemetry instrumentation
```

### Validator Request Flow (Push Experiment)

```
Miner sends PromptSynapse → Validator axon (port 8091)
  → ExperimentManager.route_submission() → routes to correct experiment
  → Experiment handler:
      1. Generate LLM response (call_chat_api_with_fallback)
      2. Moderate content (ModerationProvider)
      3. Check novelty (NoveltyClient)
      4. Score submission
  → Return scored PromptSynapse to miner
  → Validator periodically sets on-chain weights via calculate_merged_weights()
```

### Multi-Experiment Framework

Experiments are independent scoring units with their own rate limits and weight allocations.

- **Push experiments** (default): Miners submit to validators. Subclass `PushExperiment`.
- **Pull experiments**: Validators query registered miners on a schedule. Subclass `PullExperiment`.
- **ExperimentManager** coordinates all experiments: routing, rate limiting, merged weight calculation.

Key classes in `aurelius/validator/experiments/base.py`:
- `ExperimentConfig(name, experiment_type, weight_allocation, enabled, settings)` — config via settings dict, accessed with `self.setting(key, default)` in experiments
- `ExperimentScores(scores=dict, experiment_name=str, block_height=int)` — normalized scores per hotkey

### Validator Internal API

Methods are underscore-prefixed on the Validator class:
- `_get_current_block()`, `_get_miner_info(hotkey)`, `_get_network_context(hotkey)`
- Background tasks: `self.core.background_executor.submit(fn, *args)`
- Rate limiting: `self.core.experiment_manager.check_rate_limits(hotkey, experiment_id)`
- Chat: `self.core.chat_client` with `call_chat_api_with_fallback(client, params, timeout=...)`
- Moderation result: `moderation_result.flagged` (not `is_flagged`)

## Testing

```bash
uv run pytest                                          # All unit tests
uv run pytest tests/test_experiment_routing.py          # Specific file
uv run pytest -k "test_name"                            # By name pattern
RUN_INTEGRATION_TESTS=1 uv run pytest                   # Include integration tests
RUN_E2E_TESTS=1 uv run pytest                           # Include e2e tests (need Docker)
```

**Test markers**: `@pytest.mark.integration` (needs collector API), `@pytest.mark.e2e` (needs Docker/blockchain). E2e tests in `tests/e2e/` are auto-skipped unless `RUN_E2E_TESTS=1`.

**Scoring tests must use temp directories** — scoring systems persist to disk by default:
```python
import tempfile
tmp_dir = tempfile.mkdtemp()
config = ExperimentConfig(settings={"persistence_path": f"{tmp_dir}/scores.json"})
```

**Known pre-existing failures**: `test_miner_registration.py` (ModuleNotFoundError), `test_novelty_client.py`, `test_per_experiment_rate_limiter.py`, e2e tests (need Docker).

## Environment

Required (only 3 env vars):
```bash
CHUTES_API_KEY=your-key       # From chutes.ai (primary LLM)
OPENAI_API_KEY=sk-your-key    # Fallback LLM + moderation
BT_NETWORK=finney             # or 'test' for testnet, 'local' for dev
```

| Setting | Mainnet (37) | Testnet (290) | Local (1) |
|---------|-------------|---------------|-----------|
| Danger Threshold | 0.3 | 0.2 | 0.1 |
| Min Stake | 1000 TAO | 100 TAO | 0 |

## Code Style

- Python 3.10+, line length 120 (ruff)
- Lint rules: E, W, F, I (isort), N (naming), UP (pyupgrade), B (bugbear), C4 (comprehensions)
- `asyncio_mode = "auto"` in pytest (no need for `@pytest.mark.asyncio`)

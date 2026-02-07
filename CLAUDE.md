# Aurelius-Protocol Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-02-04

## Active Technologies
- Python 3.10+ (existing codebase standard) + bittensor >=6.9.0, openai >=1.50.0, pydantic, requests (all existing) (002-moral-reasoning-experiment)
- Local JSONL files for audit trail (existing `DatasetLogger` pattern), central API for aggregation (002-moral-reasoning-experiment)

- Python 3.10+ (existing codebase standard) + bittensor >=6.9.0, openai >=1.50.0, requests, pydantic (existing) (001-experiment-framework)

## Project Structure

```text
src/
tests/
```

## Commands

cd src [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] pytest [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] ruff check .

## Code Style

Python 3.10+ (existing codebase standard): Follow standard conventions

## Recent Changes
- 002-moral-reasoning-experiment: Added Python 3.10+ (existing codebase standard) + bittensor >=6.9.0, openai >=1.50.0, pydantic, requests (all existing)

- 001-experiment-framework: Added Python 3.10+ (existing codebase standard) + bittensor >=6.9.0, openai >=1.50.0, requests, pydantic (existing)

<!-- MANUAL ADDITIONS START -->

## Multi-Experiment Framework

The validator supports multiple concurrent experiments with independent scoring, rate limiting, and reward allocation.

### Key Components

- **ExperimentClient** (`aurelius/shared/experiment_client.py`): Syncs experiment definitions from central API
- **ExperimentManager** (`aurelius/validator/experiments/manager.py`): Routes submissions, manages rate limits, calculates merged weights
- **PromptSynapse** (`aurelius/shared/protocol.py`): Extended with `experiment_id`, `registration_required`, `available_experiments`

### Experiment Types

1. **Push experiments** (default): Miners submit prompts to validators
2. **Pull experiments**: Validators query registered miners on a schedule

### Adding a New Experiment

1. Define experiment in central API (or use local cache during development)
2. Experiments sync automatically every 5 minutes
3. Miners target experiments via `experiment_id` field in PromptSynapse

### Miner Registration

Miners register for non-default experiments via CLI:
```bash
python miner.py --register-experiment <experiment_id>
python miner.py --list-registrations
python miner.py --withdraw-experiment <experiment_id>
```

### Per-Experiment Novelty

Each experiment maintains an independent novelty pool. Set `experiment_id` when calling `NoveltyClient.check_novelty()`.

### Rate Limiting

Rate limits are tracked per (hotkey, experiment_id) pair. Use `ExperimentManager.check_rate_limits()`.

### Testing

```bash
pytest tests/test_experiment_client.py      # Client tests
pytest tests/test_experiment_routing.py     # Routing tests
pytest tests/test_reward_allocation.py      # Weight calculation
pytest tests/integration/test_multi_experiment.py  # Integration
```

<!-- MANUAL ADDITIONS END -->

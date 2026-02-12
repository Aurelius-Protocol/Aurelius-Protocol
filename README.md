# Aurelius Protocol

Bittensor subnet for AI alignment research. Miners submit prompts that explore moral reasoning and content boundaries; validators generate LLM responses, evaluate submissions across multiple dimensions, and score results.

| Network | Subnet | Status |
|---------|--------|--------|
| Mainnet (Finney) | 37 | Active |
| Testnet | 290 | Active |

---

## Quick Start (Validator)

```bash
# 1. Clone and install
git clone https://github.com/Aurelius-Protocol/Aurelius-Protocol.git
cd Aurelius-Protocol
pip install -e .

# 2. Configure (minimal setup)
cp .env.example .env
# Edit .env: add CHUTES_API_KEY and OPENAI_API_KEY, set BT_NETWORK=finney

# 3. Run
python validator.py
```

Wallet is auto-detected if you have exactly one wallet with one hotkey. If you have multiple wallets, add to `.env`:

```bash
VALIDATOR_WALLET_NAME=your-wallet-name
VALIDATOR_HOTKEY=your-hotkey-name
```

---

## Table of Contents

1. [Validator Guide](#validator-guide)
2. [Mining](#mining)
3. [Multi-Experiment Framework](#multi-experiment-framework)
4. [Configuration](#configuration)
5. [Troubleshooting](#troubleshooting)
6. [Security](#security)

---

## Validator Guide

### Prerequisites

#### System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| Python | 3.10+ | 3.12 |
| CPU | 2 cores | 4 cores |
| RAM | 4 GB | 8 GB |
| Storage | 20 GB | 50 GB |
| Network | 100 Mbps | 1 Gbps |

#### Required Environment Variables

Only 3 variables are required. Everything else is auto-configured.

```bash
CHUTES_API_KEY=your-key      # Get from https://chutes.ai
OPENAI_API_KEY=sk-your-key   # Get from https://platform.openai.com/api-keys (must start with sk-)
BT_NETWORK=finney            # or 'test' for testnet
```

#### Bittensor Wallet

You need a registered validator with stake on the subnet.

```bash
# Create wallet (if needed)
btcli wallet new_coldkey --wallet.name validator
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default

# Register on subnet (mainnet)
btcli subnet register --netuid 37 --wallet.name validator --subtensor.network finney

# Or register on testnet
btcli subnet register --netuid 290 --wallet.name validator --subtensor.network test
```

**Stake requirements:**
- Mainnet (subnet 37): 1000 TAO minimum
- Testnet (subnet 290): 5 TAO minimum

**Wallet location:** `~/.bittensor/wallets/`

**Wallet configuration in .env:**

The validator auto-detects your wallet if you have exactly one wallet with one hotkey. If auto-detection fails (multiple wallets, non-standard names, or custom paths), configure manually:

```bash
# Add to .env if wallet is not auto-detected
VALIDATOR_WALLET_NAME=validator    # Name of your wallet directory
VALIDATOR_HOTKEY=default           # Name of your hotkey
```

To check your wallet names:
```bash
ls ~/.bittensor/wallets/                        # List wallet names
ls ~/.bittensor/wallets/your-wallet/hotkeys/    # List hotkey names
```

---

### Network & Firewall

#### Inbound Ports (REQUIRED)

Your validator must be reachable by miners on port 8091.

| Port | Protocol | Source | Purpose |
|------|----------|--------|---------|
| 8091 | TCP | 0.0.0.0/0 (Internet) | Bittensor Axon - miners send prompts here |

#### Outbound Connections (REQUIRED)

Your validator must reach these external services over HTTPS (port 443).

| Destination | Purpose |
|-------------|---------|
| api.openai.com | OpenAI Moderation API |
| llm.chutes.ai | Chutes LLM API |
| collector.aureliusaligned.ai | Aurelius Central API (mainnet) |
| *.opentensor.ai | Bittensor subtensor RPC |
| api.ipify.org, ifconfig.me | External IP detection |

#### Firewall Setup (Linux)

```bash
# UFW (Ubuntu/Debian)
sudo ufw allow 8091/tcp comment "Aurelius Validator"
sudo ufw enable

# iptables
sudo iptables -A INPUT -p tcp --dport 8091 -j ACCEPT
```

---

### Running the Validator

#### Native

```bash
python validator.py
```

#### Docker Compose (Recommended)

```bash
# Start
docker compose up -d

# View logs
docker compose logs -f

# Restart
docker compose restart

# Update
docker compose pull && docker compose up -d

# Stop
docker compose down
```

**Docker requirements:** Docker 20.10+ and Docker Compose v2.0+

#### Docker (Manual)

```bash
docker run -d \
  --name aurelius-validator \
  --restart unless-stopped \
  -p 8091:8091 \
  --env-file .env \
  -v ~/.bittensor/wallets:/root/.bittensor/wallets:ro \
  -v aurelius-data:/var/lib/aurelius \
  ghcr.io/aurelius-protocol/aurelius-validator:latest
```

#### Systemd Service

```bash
# Create service file
sudo tee /etc/systemd/system/aurelius-validator.service << EOF
[Unit]
Description=Aurelius Validator
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
EnvironmentFile=$(pwd)/.env
ExecStart=$(which python) validator.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable aurelius-validator
sudo systemctl start aurelius-validator

# View logs
journalctl -u aurelius-validator -f
```

---

### What Validators Do

1. **Receive submissions** from miners via Bittensor axon (port 8091) or query miners directly
2. **Run experiment-specific processing** — for moral reasoning: generate a first-person LLM response to the scenario, then evaluate it across 22 signals using an AI judge
3. **Score submissions** using experiment scoring systems (danger score, novelty, hit rate)
4. **Set weights** on-chain — top 3 miners receive rewards (weighted proportional to contribution), 75% burned to UID 200

---

### Monitoring & Logs

```bash
# Docker
docker compose logs -f
docker compose logs --tail=100

# Systemd
journalctl -u aurelius-validator -f
journalctl -u aurelius-validator --since "1 hour ago"
```

**Startup validation:** The validator checks on startup that:
- OpenAI API key is valid (starts with `sk-`)
- Chutes API key is present
- Wallet is detected/configured
- Port 8091 is available

---

### Data Persistence

The validator stores data in these locations:

| Path | Purpose |
|------|---------|
| `/var/lib/aurelius/datasets/` | Dataset logs (JSONL) |
| `./miner_scores.json` | Miner scoring data |
| `./validator_trust.json` | Validator trust tracking |
| `./experiments_cache.json` | Experiment definitions cache |

**Docker volumes:**
```yaml
volumes:
  - aurelius-data:/var/lib/aurelius
  - ~/.bittensor/wallets:/root/.bittensor/wallets:ro
```

---

## Mining

### Prerequisites
- Python 3.10+
- Bittensor wallet registered on subnet

### Setup

```bash
# Install
pip install -e .

# Create wallet (if needed)
btcli wallet new_coldkey --wallet.name miner
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default

# Register on subnet
btcli subnet register --netuid 37 --wallet.name miner --subtensor.network finney
```

### Running

```bash
# Default: multi-validator mode, moral-reasoning experiment
python miner.py --prompt "Your prompt here"

# Target a specific experiment
python miner.py --prompt "Your prompt here" --experiment moral-reasoning

# Single validator
python miner.py --prompt "Your prompt here" --validator-uid 1

# Submit only (get token, don't wait for result)
python miner.py --prompt "Your prompt here" --submit-only

# With model options
python miner.py --prompt "Your prompt here" --model gpt-4o --vendor openai --temperature 0.7
```

### Async Submission Flow

Submissions are processed asynchronously via a token-based flow:

1. **Submit** — Miner sends prompt to validator, receives a submission token (~1s)
2. **Poll** — Miner polls the validator for results (every 5s by default, up to 300s)
3. **Result** — Response includes experiment-specific scores (danger score, novelty, etc.)

Use `--submit-only` to get the token without waiting, or adjust polling with `--poll-interval` and `--max-poll-time`.

### Miner Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--prompt` | Text prompt to send (required) | — |
| `--experiment` | Target experiment ID | None (validator default) |
| `--validator-uid` | Query specific validator UID | — |
| `--single` | Force single-validator mode | false |
| `--max-validators` | Max validators in multi mode | 10 |
| `--min-stake` | Min validator stake filter | 0 |
| `--netuid` | Override subnet UID | auto from BT_NETWORK |
| `--vendor` | AI vendor (chutes, openai) | None |
| `--model` | Model name | None (validator uses DeepSeek-V3.2-TEE) |
| `--temperature` | Sampling temperature (0.0-2.0) | None |
| `--top-p` | Nucleus sampling (0.0-1.0) | None |
| `--frequency-penalty` | Frequency penalty (-2.0 to 2.0) | None |
| `--presence-penalty` | Presence penalty (-2.0 to 2.0) | None |
| `--min-chars` | Min response length | None |
| `--max-chars` | Max response length | None |
| `--timeout` | Query timeout seconds | 30 |
| `--retries` | Max retry attempts | 3 |
| `--no-preflight` | Skip pre-flight health checks | false |
| `--no-color` | Disable colored output | false |
| `--poll-interval` | Seconds between status polls | 5 |
| `--max-poll-time` | Max seconds to poll | 300 |
| `--submit-only` | Submit and print token only | false |

### Multi-Validator Mode

By default, the miner queries multiple validators in parallel for faster results and redundancy.

**How it works:**
1. **Validator Discovery**: Finds validators on the subnet, sorted by stake
2. **Pre-flight Health Checks**: Tests TCP connectivity before querying (can skip with `--no-preflight`)
3. **Parallel Queries**: Sends prompts to all healthy validators simultaneously
4. **Per-validator Results**: Shows success/failure status for each validator

**Example output:**
```
Querying 5 validators: [29, 23, 45, 89, 101]...

Pre-flight health check:
  ✓ UID 29 reachable
  ✓ UID 23 reachable
  ✗ UID 45 unreachable
  ✓ UID 89 reachable
  ✓ UID 101 reachable
Health check: 4/5 validators reachable

Querying 4 validators...
[████████████████████] 4/4 complete

Results: 3/4 validators responded successfully

Failed validators:
  UID 89: No response (timeout)

Successful responses:
  UID 29: danger_score=0.45, model=deepseek-ai/DeepSeek-V3.2-TEE
  UID 23: danger_score=0.52, model=deepseek-ai/DeepSeek-V3.2-TEE
  UID 101: danger_score=0.48, model=deepseek-ai/DeepSeek-V3.2-TEE
```

### How Miners Are Scored

**Formula:** `score = severity_avg x novelty_avg ^ NOVELTY_WEIGHT`

- `severity_avg` — average danger score of accepted submissions in the current window
- `novelty_avg` — average novelty score from the central collector API
- `NOVELTY_WEIGHT` — exponent (default 1.0, i.e. linear)

**Filters** (must pass all to receive rewards):
- Hit rate >= 30% (acceptance rate of submissions)
- Average novelty >= threshold (mainnet: 0.02, testnet: 0.3)
- Max 100 submissions per window per miner

**Rewards:**
- Top 3 miners by score receive rewards, weighted proportional to their contribution
- 75% of rewards burned to UID 200

### Experiment Registration

```bash
# Register for an experiment (required for non-default experiments)
python miner.py --register-experiment jailbreak-v1

# List your registrations
python miner.py --list-registrations

# Withdraw from an experiment
python miner.py --withdraw-experiment jailbreak-v1
```

---

## Multi-Experiment Framework

The subnet supports multiple concurrent experiments with independent scoring and reward allocation.

### Current Experiments

| Experiment | Status | Description |
|------------|--------|-------------|
| `moral-reasoning` | **Active** | First-person moral dilemma responses judged across 22 signals. Miners submit scenarios; the validator generates a first-person response via LLM, then an AI judge evaluates across dimensions (virtue, care, duty, etc.) |
| `prompt` | Disabled | Dangerous prompt discovery via OpenAI moderation scoring. Not currently accepting submissions. |

### For Miners

Default behavior sends submissions to the active experiment (`moral-reasoning`). Use `--experiment <id>` to target a specific experiment:

```bash
python miner.py --prompt "Your prompt here" --experiment moral-reasoning
```

### For Validators

Experiments sync automatically from the central API. No configuration needed.

### Reward Allocation

Managed centrally. Currently: 100% to `moral-reasoning`. The `prompt` experiment is disabled and receives no allocation.

---

## Configuration

Only 3 environment variables are required:

```bash
CHUTES_API_KEY=your-key      # Chutes.ai API key
OPENAI_API_KEY=sk-your-key   # OpenAI API key (must start with sk-)
BT_NETWORK=finney            # Network: finney (mainnet) or test (testnet)
```

Everything else is auto-configured based on the network.

### Network Defaults

| Setting | Mainnet (37) | Testnet (290) |
|---------|-------------|---------------|
| Min Validator Stake | 1000 TAO | 5 TAO |
| Danger Threshold | 0.3 | 0.2 |
| Single Category Threshold | 0.7 | 0.8 |
| Min Hit Rate | 30% | 30% |
| Min Novelty Threshold | 0.02 | 0.3 |
| Burn Percentage | 75% | 75% |
| Default Model | DeepSeek-V3.2-TEE | DeepSeek-V3.2-TEE |

For advanced configuration options, see `aurelius/shared/config.py`.

---

## Troubleshooting

### Validator Issues

| Issue | Solution |
|-------|----------|
| "No wallet found" | Set `VALIDATOR_WALLET_NAME` and `VALIDATOR_HOTKEY` in .env |
| "Invalid API key" | Check OPENAI_API_KEY starts with `sk-` |
| Weight setting fails | Ensure validator is registered and has sufficient stake |
| Miners can't connect | Check firewall allows port 8091 inbound |
| "Custom error: 12" | Normal - axon already registered, harmless |

### Miner Issues

| Issue | Solution |
|-------|----------|
| "Could not find validator" | Check `BT_NETWORK` is set correctly (`finney` or `test`) |
| Connection timeout | Validator may be offline, try different UID |
| "Failed to load wallet" | Check wallet exists in `~/.bittensor/wallets/` |

---

## Security

- Never commit `.env` files
- Back up wallet mnemonics securely
- Use separate wallets for testnet/mainnet
- Set file permissions: `chmod 600 .env`
- Mount wallets read-only in Docker: `-v ~/.bittensor/wallets:/root/.bittensor/wallets:ro`

---

## Support

- GitHub Issues: https://github.com/Aurelius-Protocol/Aurelius-Protocol/issues
- Bittensor Discord: https://discord.gg/bittensor

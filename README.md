# Aurelius Protocol

Bittensor subnet for AI alignment research. Miners discover prompts that trigger content moderation systems while remaining ethical; validators process and score submissions.

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
- Testnet (subnet 290): 100 TAO minimum

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

1. **Receive prompts** from miners via Bittensor axon (port 8091)
2. **Generate responses** using configured LLM (Chutes/DeepSeek by default)
3. **Moderate content** using OpenAI moderation API
4. **Check novelty** against previous submissions
5. **Score miners** based on danger, novelty, and hit rate
6. **Set weights** on-chain for miner rewards

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
# Multi-validator mode (default) - queries top validators by stake
python miner.py --prompt "Your prompt here"

# Limit number of validators
python miner.py --prompt "Your prompt here" --max-validators 5

# Query a single specific validator
python miner.py --prompt "Your prompt here" --validator-uid 1

# With model options
python miner.py \
  --prompt "Explain quantum computing" \
  --model deepseek-ai/DeepSeek-V3 \
  --temperature 0.7
```

### Miner Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--prompt` | Text prompt to send (required) | - |
| `--validator-uid` | Query specific validator UID (single mode) | - |
| `--max-validators` | Max validators to query in multi mode | 10 |
| `--min-stake` | Minimum stake for validator selection | 1000.0 |
| `--no-preflight` | Skip pre-flight health checks | false |
| `--model` | Model (deepseek-ai/DeepSeek-V3, gpt-4o, etc.) | deepseek-ai/DeepSeek-V3 |
| `--temperature` | Sampling temperature (0.0-2.0) | 0.7 |

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
  UID 29: danger_score=0.45, model=deepseek-ai/DeepSeek-V3
  UID 23: danger_score=0.52, model=deepseek-ai/DeepSeek-V3
  UID 101: danger_score=0.48, model=deepseek-ai/DeepSeek-V3
```

### How Miners Are Scored

Miners are scored based on:
1. **Danger Score** - How effectively the prompt triggers moderation (0-1)
2. **Novelty Score** - Uniqueness compared to previous submissions (0-1)
3. **Hit Rate** - Percentage of accepted submissions

Weight formula: `danger_sum × severity_avg × novelty_multiplier`

Top 3 miners per window receive rewards, split by contribution.

---

## Multi-Experiment Framework

The subnet supports multiple concurrent experiments with independent scoring and reward allocation.

### For Miners

**Default behavior (no changes needed):** Submissions go to the "prompt" experiment.

**Targeting a specific experiment:**
```python
synapse = PromptSynapse(
    prompt="Your prompt",
    experiment_id="jailbreak-v1"  # Target experiment
)
```

**Experiment registration:**
```bash
# Register for an experiment (required for non-default experiments)
python miner.py --register-experiment jailbreak-v1

# List your registrations
python miner.py --list-registrations

# Withdraw from an experiment
python miner.py --withdraw-experiment jailbreak-v1
```

### For Validators

Experiments sync automatically from the central API every 5 minutes. No configuration needed.

**Optional environment variables:**
```bash
EXPERIMENT_SYNC_INTERVAL=300              # Sync interval in seconds
EXPERIMENT_CACHE_PATH=./experiments_cache.json  # Cache file path
```

### Reward Allocation

Rewards are distributed across experiments based on configured allocation percentages:
- Default: 85% to "prompt" experiment
- Additional experiments share remaining allocation
- Unused allocation redistributes proportionally

See `specs/001-experiment-framework/quickstart.md` for detailed documentation.

---

## Configuration

Only 3 environment variables are required:

```bash
CHUTES_API_KEY=your-key      # Chutes.ai API key
OPENAI_API_KEY=sk-your-key   # OpenAI API key (must start with sk-)
BT_NETWORK=finney            # Network: finney (mainnet) or test (testnet)
```

Everything else is auto-configured based on the network.

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
| "Could not find validator" | Check `BT_NETUID` and `BT_NETWORK` match |
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

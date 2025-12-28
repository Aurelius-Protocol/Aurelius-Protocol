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

That's it. Wallet detection, network defaults, and telemetry are automatic.

---

## Table of Contents

1. [Mining](#mining)
2. [Validation](#validation)
3. [Configuration](#configuration)
4. [Deployment](#deployment)
5. [Troubleshooting](#troubleshooting)

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
# Basic usage
python miner.py --prompt "Your prompt here" --validator-uid 1

# With options
python miner.py \
  --prompt "Explain quantum computing" \
  --model deepseek-ai/DeepSeek-V3 \
  --temperature 0.7 \
  --validator-uid 1
```

### Miner Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--prompt` | Text prompt to send (required) | - |
| `--validator-uid` | Validator UID to query | 1 |
| `--model` | Model (deepseek-ai/DeepSeek-V3, gpt-4o, etc.) | deepseek-ai/DeepSeek-V3 |
| `--temperature` | Sampling temperature (0.0-2.0) | 0.7 |

### How Miners Are Scored

Miners are scored based on:
1. **Danger Score** - How effectively the prompt triggers moderation (0-1)
2. **Novelty Score** - Uniqueness compared to previous submissions (0-1)
3. **Hit Rate** - Percentage of accepted submissions

Weight formula: `danger_sum × severity_avg × novelty_multiplier`

Top 3 miners per window receive rewards, split by contribution.

---

## Validation

### Prerequisites
- Python 3.10+
- OpenAI API key (required for moderation)
- Chutes API key (for chat completions)
- Registered validator with stake

### Minimal Setup (Turnkey)

```bash
# 1. Copy minimal config
cp .env.example .env

# 2. Edit .env with only these values:
CHUTES_API_KEY=your-chutes-key
OPENAI_API_KEY=sk-your-openai-key
BT_NETWORK=finney  # or 'test' for testnet
```

Everything else is auto-detected:
- Wallet (if you have exactly one)
- Network defaults (thresholds, stake requirements)
- External IP (with `AUTO_DETECT_EXTERNAL_IP=true`)
- Telemetry (enabled by default)

### Running

**Native:**
```bash
python validator.py
```

**Docker Compose (recommended):**
```bash
docker compose up -d
docker compose logs -f  # View logs
```

**Docker (manual):**
```bash
docker run -d \
  --name aurelius-validator \
  --restart unless-stopped \
  -p 8091:8091 \
  --env-file .env \
  -v ~/.bittensor/wallets:/root/.bittensor/wallets:ro \
  ghcr.io/aurelius-protocol/aurelius-validator:latest
```

### What Validators Do

1. **Receive prompts** from miners via Bittensor axon
2. **Generate responses** using configured LLM (Chutes/DeepSeek by default)
3. **Moderate content** using OpenAI moderation API
4. **Check novelty** against previous submissions
5. **Score miners** based on danger, novelty, and hit rate
6. **Set weights** on-chain for miner rewards

---

## Configuration

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `CHUTES_API_KEY` | Chutes.ai API key for chat completions |
| `OPENAI_API_KEY` | OpenAI API key for moderation (must start with `sk-`) |
| `BT_NETWORK` | Network: `finney` (mainnet) or `test` (testnet) |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VALIDATOR_WALLET_NAME` | Auto-detect | Wallet name |
| `VALIDATOR_HOTKEY` | Auto-detect | Hotkey name |
| `AUTO_DETECT_EXTERNAL_IP` | `true` | Auto-detect external IP for axon |
| `VALIDATOR_PORT` | `8091` | Axon port |
| `TELEMETRY_ENABLED` | `true` | Send telemetry to Aurelius collector |
| `ADVANCED_MODE` | `false` | Use custom thresholds instead of network defaults |

### Network Defaults

These are applied automatically based on `BT_NETWORK`. Do not override unless using `ADVANCED_MODE=true`.

| Setting | Testnet (290) | Mainnet (37) |
|---------|---------------|--------------|
| `DANGER_THRESHOLD` | 0.2 | 0.3 |
| `SINGLE_CATEGORY_THRESHOLD` | 0.8 | 0.7 |
| `MIN_HIT_RATE_THRESHOLD` | 0.3 | 0.3 |
| `MIN_NOVELTY_THRESHOLD` | 0.3 | 0.02 |
| `MIN_VALIDATOR_STAKE` | 100 TAO | 1000 TAO |

### Advanced Mode

By default (`ADVANCED_MODE=false`), network-specific defaults override any values in your `.env`. This ensures all validators use consistent parameters.

To use custom thresholds:
```bash
ADVANCED_MODE=true
DANGER_THRESHOLD=0.4  # Now this will be used
```

---

## Deployment

### Server Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 2 cores | 4 cores |
| RAM | 4 GB | 8 GB |
| Storage | 20 GB | 50 GB |
| Network | 100 Mbps | 1 Gbps |

### Firewall

```bash
# Required ports
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 8091/tcp  # Bittensor axon
sudo ufw enable
```

### Docker Compose (Recommended)

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

### Native with Systemd

```bash
# Install
pip install -e .

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

## Troubleshooting

### Validator Issues

| Issue | Solution |
|-------|----------|
| "No wallet found" | Set `VALIDATOR_WALLET_NAME` and `VALIDATOR_HOTKEY` in .env |
| "Invalid API key" | Check OPENAI_API_KEY starts with `sk-` |
| Weight setting fails | Ensure validator is registered and has sufficient stake |
| Miners can't connect | Check firewall allows port 8091, set `AUTO_DETECT_EXTERNAL_IP=true` |
| "Custom error: 12" | Normal - axon already registered, harmless |

### Miner Issues

| Issue | Solution |
|-------|----------|
| "Could not find validator" | Check `BT_NETUID` and `BT_NETWORK` match |
| Connection timeout | Validator may be offline, try different UID |
| "Failed to load wallet" | Check wallet exists in `~/.bittensor/wallets/` |

### Logs

```bash
# Docker
docker compose logs -f
docker compose logs --tail=100

# Native/Systemd
journalctl -u aurelius-validator -f
journalctl -u aurelius-validator --since "1 hour ago"
```

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

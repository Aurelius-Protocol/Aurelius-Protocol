# Aurelius Miner

Miner for the Aurelius subnet on Bittensor. Submit prompts to validators for OpenAI processing and content moderation.

## Overview

The Aurelius miner allows you to submit text prompts to validators on the Aurelius subnet. Validators process these prompts using OpenAI's API and perform content moderation, returning both the AI-generated response and safety scores.

## Features

- ✅ Submit prompts to validators on the Aurelius subnet
- ✅ Receive OpenAI-generated responses with moderation scores
- ✅ **Model specification** - Request specific AI models (gpt-4o, gpt-4o-mini, etc.)
- ✅ **Parameter customization** - Control temperature, top_p, penalties, and response length
- ✅ Support for mainnet (finney), testnet, and local development
- ✅ Flexible configuration via environment variables
- ✅ Simple command-line interface

## Installation

### Prerequisites

- Python 3.10 or higher
- A Bittensor wallet with registered hotkey
- (Optional) TAO tokens for mainnet operation

### 1. Clone the Repository

```bash
git clone https://github.com/Aurelius-Protocol/Aurelius-Protocol.git
cd Aurelius-Protocol
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the miner package
pip install -e .
```

### 3. Create Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your preferred editor
nano .env  # or vim, code, etc.
```

Update the following in your `.env` file:
- `MINER_WALLET_NAME`: Your wallet name (default: "miner")
- `MINER_HOTKEY`: Your hotkey name (default: "default")
- `BT_NETWORK`: Network to use ("finney" for mainnet, "test" for testnet)
- `BT_NETUID`: Subnet UID for Aurelius

## Wallet Setup

### Create a New Wallet

If you don't have a Bittensor wallet yet:

```bash
# Create a coldkey (stores your TAO)
btcli wallet new_coldkey --wallet.name miner

# Create a hotkey (used for subnet operations)
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
```

Your wallet files will be stored in `~/.bittensor/wallets/miner/`

⚠️ **IMPORTANT**: Back up your coldkey mnemonic phrase in a secure location!

### Register on the Subnet

To participate on mainnet or testnet, you must register your hotkey:

```bash
# For mainnet
btcli subnet register --netuid 1 --wallet.name miner --wallet.hotkey default

# For testnet
btcli subnet register --netuid 37 --wallet.name miner --wallet.hotkey default --subtensor.network test
```

Note: Registration requires a small amount of TAO for the transaction fee.

## Usage

### Basic Usage

Submit a prompt to a validator:

```bash
python miner.py --prompt "Explain how photosynthesis works" --validator-uid 1
```

### Using the Module Directly

```bash
python -m aurelius.miner.miner --prompt "Write a Python hello world script" --validator-uid 1
```

### After Installation with pip

If you installed with `pip install -e .`, you can use the command:

```bash
aurelius-miner --prompt "Your prompt here" --validator-uid 1
```

### Command-Line Arguments

**Required:**
- `--prompt`: The text prompt to send to the validator

**Connection:**
- `--validator-uid` (default=1): UID of the validator to query
- `--netuid`: Override the subnet UID from config

**Model Specification:**
- `--vendor`: AI vendor to use (e.g., `openai`, `anthropic`)
- `--model`: Specific model to use (e.g., `gpt-4o`, `gpt-4o-mini`, `o4-mini`)
- `--temperature`: Sampling temperature (0.0-2.0, higher = more creative)
- `--top-p`: Nucleus sampling parameter (0.0-1.0)
- `--frequency-penalty`: Reduce repetition of frequent tokens (-2.0 to 2.0)
- `--presence-penalty`: Reduce repetition of any mentioned topics (-2.0 to 2.0)
- `--min-chars`: Minimum response length in characters
- `--max-chars`: Maximum response length in characters

### Example Prompts

```bash
# Basic prompt (uses validator defaults)
python miner.py --prompt "Explain how machine learning models work" --validator-uid 1

# Request a specific model
python miner.py --prompt "Write a technical analysis" --model gpt-4o

# Creative writing with high temperature
python miner.py --prompt "Write a short story about a robot" --temperature 1.2

# Controlled response with custom parameters
python miner.py --prompt "Explain quantum computing" \
  --model gpt-4o \
  --temperature 0.7 \
  --top-p 0.9 \
  --min-chars 100 \
  --max-chars 2000

# Full customization example
python miner.py --prompt "Write a product description" \
  --vendor openai \
  --model gpt-4o \
  --temperature 0.8 \
  --top-p 0.95 \
  --frequency-penalty 0.3 \
  --presence-penalty 0.2 \
  --min-chars 50 \
  --max-chars 1500
```

## Configuration

All configuration is done through environment variables in the `.env` file:

### Required Settings

| Variable | Description | Example |
|----------|-------------|---------|
| `MINER_WALLET_NAME` | Your Bittensor wallet name | `miner` |
| `MINER_HOTKEY` | Your wallet hotkey name | `default` |
| `BT_NETWORK` | Network to connect to | `finney`, `test`, or `local` |
| `BT_NETUID` | Aurelius subnet UID | `1` |

### Optional Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `BT_PORT_VALIDATOR` | Validator port | `8091` |
| `SUBTENSOR_ENDPOINT` | Custom subtensor endpoint | (uses network default) |
| `LOCAL_MODE` | Skip blockchain, connect by IP | `false` |
| `VALIDATOR_HOST` | Validator IP (LOCAL_MODE only) | `127.0.0.1` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `LOG_SENSITIVE_DATA` | Log full prompts/responses | `false` |
| `MAX_LOG_LENGTH` | Max chars to log | `100` |

### Local Mode (Development)

For testing without blockchain registration:

```bash
# In .env
LOCAL_MODE=true
VALIDATOR_HOST=46.62.225.78  # IP of a test validator
```

This bypasses blockchain lookups and connects directly to the specified validator IP.

## Model Specification

The miner supports requesting specific AI models and customizing generation parameters. The validator will respect your preferences when possible.

### How It Works

1. **Miner requests** a specific model and parameters via command-line arguments
2. **Validator validates** the request against allowed models and parameter ranges
3. **Validator uses** the requested configuration (or falls back to defaults if invalid)
4. **Response includes** both the requested and actual values used

### Available Models (OpenAI)

- `gpt-4o` - Most capable model, best for complex tasks
- `gpt-4o-mini` - Faster and cheaper, good for most tasks (default)
- `o4-mini` - Optimized for reasoning tasks
- `o3-mini` - Cost-effective reasoning model
- `gpt-4-turbo` - High capability with larger context
- `gpt-3.5-turbo` - Fast and economical

### Parameter Guidelines

| Parameter | Range | Description |
|-----------|-------|-------------|
| `temperature` | 0.0 - 2.0 | Controls randomness. 0.0 = deterministic, 2.0 = very creative |
| `top_p` | 0.0 - 1.0 | Nucleus sampling. Lower values = more focused responses |
| `frequency_penalty` | -2.0 - 2.0 | Positive = reduce word repetition |
| `presence_penalty` | -2.0 - 2.0 | Positive = encourage topic diversity |

### Validation and Fallback

- If you request an invalid model, the validator uses the default (`gpt-4o-mini`)
- Parameters outside valid ranges are automatically clamped
- Character limits are enforced but may result in truncated responses
- All requested vs. actual values are logged for transparency

### Example: Research Query

```bash
python miner.py \
  --prompt "Analyze the environmental impact of electric vehicles" \
  --model gpt-4o \
  --temperature 0.3 \
  --min-chars 500 \
  --max-chars 3000
```

This requests the most capable model with low temperature for factual accuracy and specific length requirements.

## Understanding the Response

When you submit a prompt, the validator returns:

```
============================================================
RESPONSE FROM VALIDATOR
============================================================
Prompt:   Explain quantum computing
Response: Quantum computing is a type of computation that...
Model:    gpt-4o

--- Moderation Results ---
Danger Score:  0.0000
Accepted:      ✓ YES

Top Category Scores:
============================================================
```

- **Prompt**: Your original request
- **Response**: The AI-generated answer
- **Model**: The actual model used (reflects your request if valid)
- **Danger Score**: Content moderation score (0-1, higher = more dangerous)
- **Accepted**: Whether the content met safety thresholds (✓ YES or ✗ NO)
- **Top Category Scores**: Breakdown of moderation categories (if any concerns detected)

## Troubleshooting

### "Could not find validator with UID X"

- Check that the validator UID exists on the subnet
- Verify your `BT_NETUID` is correct
- Ensure you're connected to the right network (`BT_NETWORK`)

### "Failed to load wallet"

- Verify your wallet name and hotkey in `.env`
- Check that wallet files exist in `~/.bittensor/wallets/`
- Ensure correct permissions on wallet files

### "Connection timeout"

- Validator may be offline
- Try a different validator UID
- Check your internet connection
- Verify the subnet is active on your chosen network

### "No valid response received"

- The validator may be rate-limiting requests
- Try again in a few minutes
- Use a different validator UID

## Development

### Project Structure

```
Aurelius-Protocol/
├── miner.py                    # Entry point script
├── pyproject.toml              # Package configuration
├── .env.example                # Example environment config
├── README.md                   # This file
└── aurelius/                   # Main package
    ├── __init__.py
    ├── miner/
    │   ├── __init__.py
    │   └── miner.py           # Miner implementation
    └── shared/
        ├── __init__.py
        ├── protocol.py         # Synapse definitions
        └── config.py           # Configuration management
```

### Running Tests

```bash
# Test with a simple prompt
python miner.py --prompt "Test prompt" --validator-uid 69

# Test in local mode (set LOCAL_MODE=true in .env first)
python miner.py --prompt "Test prompt" --validator-uid 1
```

## Security Notes

- ⚠️ Never commit your `.env` file (it's in `.gitignore`)
- ⚠️ Never share your wallet mnemonic or coldkey files
- ⚠️ Back up your wallet files securely
- ⚠️ Use separate wallets for testnet and mainnet

## Support

- GitHub Issues: https://github.com/Aurelius-Protocol/Aurelius-Protocol/issues
- Bittensor Discord: https://discord.gg/bittensor

## License

MIT License - see LICENSE file for details

## Acknowledgments

Built on the Bittensor network and powered by OpenAI's language models.

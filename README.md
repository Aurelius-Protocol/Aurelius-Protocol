# Aurelius Miner

Miner for the Aurelius subnet on Bittensor. Submit prompts to validators for OpenAI processing and content moderation.

## Overview

The Aurelius miner allows you to submit text prompts to validators on the Aurelius subnet. Validators process these prompts using OpenAI's API and perform content moderation, returning both the AI-generated response and safety scores.

## Features

- ✅ Submit prompts to validators on the Aurelius subnet
- ✅ Receive OpenAI-generated responses with moderation scores
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

- `--prompt` (required): The text prompt to send to the validator
- `--validator-uid` (optional, default=1): UID of the validator to query
- `--netuid` (optional): Override the subnet UID from config

### Example Prompts

```bash
# Safe educational prompt
python miner.py --prompt "Explain how machine learning models work" --validator-uid 1

# Creative writing
python miner.py --prompt "Write a short story about a robot" --validator-uid 1

# Technical question
python miner.py --prompt "How do I sort a list in Python?" --validator-uid 1
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

## Understanding the Response

When you submit a prompt, the validator returns:

```
Response: [The OpenAI-generated text]
Model: gpt-4o-mini
Danger Score: 0.00032
Accepted: true
```

- **Response**: The AI-generated answer to your prompt
- **Model**: Which OpenAI model was used
- **Danger Score**: Content moderation score (0-1, higher = more dangerous)
- **Accepted**: Whether the content met safety thresholds

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

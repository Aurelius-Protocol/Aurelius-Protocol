# Aurelius Validator Deployment

Automated deployment script for deploying the Aurelius validator to a remote server.

## Prerequisites

### Local Machine
- `sshpass` installed (`sudo apt-get install sshpass`)
- `rsync` installed
- Bittensor wallet configured in `~/.bittensor/`
- OpenAI API key

### Remote Server
- Fresh Ubuntu 24.04 installation
- Root or sudo access
- Server credentials in `.passw` file

## Quick Start

### First Deployment

1. Ensure your `.passw` file contains server credentials:
```bash
sshpass -p 'your_password' ssh root@your.server.ip
```

2. Run the deployment script:
```bash
./deploy.sh
```

3. SSH to the remote server and configure the validator:
```bash
ssh root@your.server.ip
nano /opt/aurelius-validator/.env
```

4. Set the following in `.env`:
```bash
OPENAI_API_KEY=sk-your-key-here
VALIDATOR_WALLET_NAME=your_wallet_name
VALIDATOR_HOTKEY=your_hotkey_name
```

5. Restart the validator service:
```bash
sudo systemctl restart aurelius-validator
```

6. Monitor the logs:
```bash
sudo journalctl -u aurelius-validator -f
```

### Updating the Validator

To deploy updates to an existing installation:

1. Make your changes to the validator code locally in `/home/volker/code/aurelius/http-test-net`

2. Run the deployment script again:
```bash
./deploy.sh
```

The script will automatically:
- Create a timestamped backup of the current deployment
- Stop the service gracefully
- Transfer updated files
- Restart the service

## What the Script Does

### Initial Deployment
1. Parses server credentials from `.passw` file
2. Installs Python 3.11 and system dependencies
3. Configures UFW firewall (ports 22, 8091)
4. Creates deployment directories
5. Transfers validator files via rsync
6. Copies Bittensor wallet from local machine
7. Creates Python virtual environment
8. Installs Python dependencies
9. Configures `.env` from testnet template
10. Creates systemd service
11. Starts and verifies the validator

### Update Deployment
1. Detects existing installation
2. Creates backup with timestamp
3. Stops validator service
4. Transfers updated files
5. Updates dependencies if needed
6. Restarts validator service
7. Verifies operation

## Remote Server Locations

- **Validator code**: `/opt/aurelius-validator/`
- **Virtual environment**: `/opt/aurelius-validator/.venv/`
- **Configuration**: `/opt/aurelius-validator/.env`
- **Datasets**: `/var/lib/aurelius/datasets/`
- **Logs**: `/var/lib/aurelius/validator.log`
- **Error logs**: `/var/lib/aurelius/validator.error.log`
- **Wallet**: `~/.bittensor/`

## Service Management

```bash
# Check status
sudo systemctl status aurelius-validator

# Start service
sudo systemctl start aurelius-validator

# Stop service
sudo systemctl stop aurelius-validator

# Restart service
sudo systemctl restart aurelius-validator

# View logs (real-time)
sudo journalctl -u aurelius-validator -f

# View recent logs
sudo journalctl -u aurelius-validator -n 100
```

## Troubleshooting

### Service won't start

Check the logs:
```bash
sudo journalctl -u aurelius-validator -n 50
```

Common issues:
- Missing or invalid OPENAI_API_KEY in `.env`
- Incorrect wallet configuration
- Wallet not registered on subnet

### Connection issues

Verify firewall:
```bash
sudo ufw status
```

Check if port 8091 is open:
```bash
sudo ss -tlnp | grep 8091
```

### Wallet issues

Verify wallet permissions:
```bash
ls -la ~/.bittensor/wallets/*/hotkeys/
```

Hotkey files should be mode 600.

### Re-deploying from scratch

If you need to completely remove and re-deploy:

```bash
# On remote server
sudo systemctl stop aurelius-validator
sudo systemctl disable aurelius-validator
sudo rm /etc/systemd/system/aurelius-validator.service
sudo rm -rf /opt/aurelius-validator
sudo rm -rf /var/lib/aurelius
sudo systemctl daemon-reload

# Then run ./deploy.sh again from local machine
```

## Configuration Reference

### Key `.env` Variables for Testnet

```bash
# Network
BT_NETWORK=test
SUBTENSOR_ENDPOINT=wss://test.finney.opentensor.ai:443
BT_NETUID=1

# API Keys
OPENAI_API_KEY=sk-your-key-here

# Wallet
VALIDATOR_WALLET_NAME=validator
VALIDATOR_HOTKEY=default

# Networking
VALIDATOR_HOST=0.0.0.0
BT_PORT_VALIDATOR=8091
AUTO_DETECT_EXTERNAL_IP=true

# Data
DATASET_DIR=/var/lib/aurelius/datasets

# Operation Mode
LOCAL_MODE=false

# Security
MODERATION_FAIL_MODE=closed
LOG_SENSITIVE_DATA=false
```

## Security Notes

- The `.passw` file contains sensitive credentials - never commit it to git
- Wallet files are transferred with 600 permissions (owner read/write only)
- `.env` file should be edited on the remote server after deployment
- Consider using SSH keys instead of password authentication for production

## Python Version

This deployment script uses **Python 3.12** (native to Ubuntu 24.04), which is fully compatible with Bittensor and dependencies (requires Python >= 3.10).

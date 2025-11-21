# Aurelius Validator Deployment Summary

## ✅ Deployment Complete!

**Date:** 2025-11-11
**Server:** root@46.62.225.78
**Status:** Validator service running

---

## What Was Deployed

### ✓ System Configuration
- **OS:** Ubuntu 24.04
- **Python:** 3.12
- **Firewall:** UFW enabled (ports 22, 8091 open)
- **SSH:** Key-based authentication configured

### ✓ Validator Installation
- **Location:** `/opt/aurelius-validator/`
- **Virtual Environment:** Python 3.12 venv at `/opt/aurelius-validator/.venv/`
- **Service:** `aurelius-validator.service` (enabled, auto-start on boot)
- **Logs:**
  - Output: `/var/lib/aurelius/validator.log`
  - Errors: `/var/lib/aurelius/validator.error.log`

### ✓ Wallet
- **Location:** `/root/.bittensor/wallets/validator/`
- **Status:** Transferred successfully with coldkey and hotkeys
- **Configured Wallet:** validator
- **Configured Hotkey:** default

### ✓ Configuration
- **Network:** Testnet (`test`)
- **Subnet ID:** 1
- **API Key:** OpenAI API key configured
- **Mode:** Production (LOCAL_MODE=false)

---

## Current Status

### Service Status
```bash
systemctl status aurelius-validator
```
- **Active:** Running
- **Auto-start:** Enabled
- **Memory Usage:** ~170MB

### Known Issue
⚠️ **Bittensor Subtensor Error:**
```
SubstrateRequestException(Invalid Transaction) - Custom error: 10
```

**What this means:**
This error typically indicates one of the following:
1. **Wallet not registered on testnet subnet** - The validator wallet needs to be registered on subnet 1
2. **Insufficient stake** - The wallet may need TAO tokens staked
3. **Wallet configuration mismatch** - Double-check wallet name/hotkey

**How to resolve:**

1. Check wallet registration:
```bash
ssh root@46.62.225.78
cd /opt/aurelius-validator
source .venv/bin/activate
btcli wallet overview --wallet.name validator --wallet.hotkey default --subtensor.network test
```

2. Register wallet on subnet (if not registered):
```bash
btcli subnet register \
  --wallet.name validator \
  --wallet.hotkey default \
  --subtensor.network test \
  --netuid 1
```

3. Check if stake is required and add if needed:
```bash
btcli stake add \
  --wallet.name validator \
  --wallet.hotkey default \
  --amount X \
  --subtensor.network test
```

---

## Service Management Commands

### Check Status
```bash
ssh root@46.62.225.78 systemctl status aurelius-validator
```

### View Logs (Real-time)
```bash
ssh root@46.62.225.78 journalctl -u aurelius-validator -f
```

### View Recent Logs
```bash
ssh root@46.62.225.78 tail -100 /var/lib/aurelius/validator.log
ssh root@46.62.225.78 tail -100 /var/lib/aurelius/validator.error.log
```

### Restart Service
```bash
ssh root@46.62.225.78 systemctl restart aurelius-validator
```

### Stop Service
```bash
ssh root@46.62.225.78 systemctl stop aurelius-validator
```

### Start Service
```bash
ssh root@46.62.225.78 systemctl start aurelius-validator
```

---

## Updating the Validator

When you make changes to the validator code locally, simply run:

```bash
cd /home/volker/code/aurelius/deployment
./deploy.sh
```

The script will automatically:
1. Create a timestamped backup of the current deployment
2. Stop the validator service
3. Transfer updated files
4. Restart the service

**Note:** The `.env` file is preserved during updates, so your configuration won't be overwritten.

---

## File Locations

### On Remote Server

| Item | Location |
|------|----------|
| Validator code | `/opt/aurelius-validator/` |
| Virtual environment | `/opt/aurelius-validator/.venv/` |
| Configuration | `/opt/aurelius-validator/.env` |
| Wallet | `/root/.bittensor/wallets/` |
| Datasets | `/var/lib/aurelius/datasets/` |
| Output logs | `/var/lib/aurelius/validator.log` |
| Error logs | `/var/lib/aurelius/validator.error.log` |
| Systemd service | `/etc/systemd/system/aurelius-validator.service` |

### On Local Machine

| Item | Location |
|------|----------|
| Deployment script | `/home/volker/code/aurelius/deployment/deploy.sh` |
| Server credentials | `/home/volker/code/aurelius/deployment/.passw` |
| Validator source | `/home/volker/code/aurelius/http-test-net/` |
| SSH key | `/home/volker/.ssh/id_rsa` |

---

## Configuration Reference

### Current .env Settings

```bash
# Network Configuration
BT_NETWORK=test
BT_NETUID=1
SUBTENSOR_ENDPOINT=wss://test.finney.opentensor.ai:443

# Wallet Configuration
VALIDATOR_WALLET_NAME=validator
VALIDATOR_HOTKEY=default

# API Keys
OPENAI_API_KEY=sk-proj-****** (configured)

# Network Settings
VALIDATOR_HOST=0.0.0.0
BT_PORT_VALIDATOR=8091
AUTO_DETECT_EXTERNAL_IP=true

# Data
DATASET_DIR=/var/lib/aurelius/datasets

# Mode
LOCAL_MODE=false
```

### To Edit Configuration

1. SSH to server:
```bash
ssh root@46.62.225.78
```

2. Edit .env:
```bash
nano /opt/aurelius-validator/.env
```

3. Restart service:
```bash
systemctl restart aurelius-validator
```

---

## Security Notes

- **SSH:** Key-based authentication is enabled. Password: `rijwaw-cedhed-Nonjy4`
- **Firewall:** UFW enabled with only ports 22 (SSH) and 8091 (validator) open
- **Credentials:** Server password stored in `.passw` file (chmod 600)
- **Wallet:** Coldkey and hotkeys have restricted permissions (600)
- **API Keys:** OpenAI API key configured in `.env`

---

## Troubleshooting

### Validator Not Starting

```bash
# Check logs
ssh root@46.62.225.78 journalctl -u aurelius-validator -n 100

# Check service status
ssh root@46.62.225.78 systemctl status aurelius-validator

# Try manual start
ssh root@46.62.225.78
cd /opt/aurelius-validator
source .venv/bin/activate
python validator.py
```

### Permission Errors

```bash
# Fix ownership
ssh root@46.62.225.78 "chown -R root:root /opt/aurelius-validator /var/lib/aurelius"

# Fix wallet permissions
ssh root@46.62.225.78 "chmod 700 ~/.bittensor && chmod 600 ~/.bittensor/wallets/validator/coldkey"
```

### Network Connectivity Issues

```bash
# Check firewall
ssh root@46.62.225.78 ufw status

# Check if port is listening
ssh root@46.62.225.78 ss -tlnp | grep 8091

# Test external IP
ssh root@46.62.225.78 curl ifconfig.me
```

### Redeployment from Scratch

If something goes wrong and you need to start fresh:

```bash
# On remote server (via SSH or console)
systemctl stop aurelius-validator
systemctl disable aurelius-validator
rm /etc/systemd/system/aurelius-validator.service
rm -rf /opt/aurelius-validator
rm -rf /var/lib/aurelius
systemctl daemon-reload

# Then run deployment script again from local machine
./deploy.sh
```

---

## Next Steps

1. **Resolve Bittensor Registration:**
   - Register wallet on testnet subnet 1
   - Ensure wallet has sufficient stake if required

2. **Monitor Validator:**
   - Watch logs for successful operations
   - Verify miner connections
   - Check dataset collection

3. **Production Considerations:**
   - Consider switching to mainnet when ready
   - Monitor server resources (CPU, memory, disk)
   - Set up monitoring/alerting for service downtime
   - Regular backups of wallet coldkey (CRITICAL!)

---

## Support Resources

- **Bittensor Docs:** https://docs.bittensor.com
- **Error Reference:** https://docs.bittensor.com/errors/custom
- **Subnet Registration:** https://docs.bittensor.com/subnets/register-validate-mine

---

## Deployment Script Features

The `deploy.sh` script handles:
- ✓ Automatic SSH key generation and setup
- ✓ System dependency installation (Python 3.12, build tools)
- ✓ Firewall configuration
- ✓ File transfer (rsync with exclusions)
- ✓ Wallet transfer with proper permissions
- ✓ Python virtual environment setup
- ✓ Dependency installation
- ✓ Systemd service creation
- ✓ Automatic backups on updates
- ✓ Graceful service restarts
- ✓ Configuration preservation during updates

**First run:** Full deployment (~5-10 minutes)
**Subsequent runs:** Quick updates (~1-2 minutes)

#!/bin/bash

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATOR_SOURCE_DIR="/home/volker/Code/aurelius/Aurelius-Protocol"
REMOTE_DEPLOY_DIR="/opt/aurelius-validator"
REMOTE_DATA_DIR="/var/lib/aurelius"
WALLET_SOURCE_DIR="$HOME/.bittensor"
PYTHON_VERSION="3.12"
SSH_KEY_PATH="$HOME/.ssh/id_rsa"

# Parse server credentials from .passw file
parse_credentials() {
    echo -e "${BLUE}Parsing server credentials...${NC}"

    if [[ ! -f "$SCRIPT_DIR/.passw" ]]; then
        echo -e "${RED}Error: .passw file not found at $SCRIPT_DIR/.passw${NC}"
        exit 1
    fi

    # Parse: sshpass -p 'password' ssh user@host
    local passw_content=$(cat "$SCRIPT_DIR/.passw")

    SERVER_PASSWORD=$(echo "$passw_content" | grep -oP "(?<=-p ')[^']+")
    SERVER_USER=$(echo "$passw_content" | grep -oP "ssh \K[^@]+(?=@)")
    SERVER_HOST=$(echo "$passw_content" | grep -oP "@\K[0-9.]+")

    if [[ -z "$SERVER_PASSWORD" ]] || [[ -z "$SERVER_USER" ]] || [[ -z "$SERVER_HOST" ]]; then
        echo -e "${RED}Error: Failed to parse credentials from .passw file${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Credentials parsed: ${SERVER_USER}@${SERVER_HOST}${NC}"
}

# Check if SSH key exists, generate if needed
setup_ssh_key() {
    echo -e "${BLUE}Setting up SSH key authentication...${NC}"

    # Check if SSH key exists
    if [[ ! -f "$SSH_KEY_PATH" ]]; then
        echo -e "${YELLOW}SSH key not found, generating new key...${NC}"
        ssh-keygen -t rsa -b 4096 -f "$SSH_KEY_PATH" -N "" -C "aurelius-deployment"
        echo -e "${GREEN}✓ SSH key generated${NC}"
    else
        echo -e "${GREEN}✓ SSH key exists${NC}"
    fi

    # Test if key-based auth already works
    if ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$SERVER_USER@$SERVER_HOST" "exit" 2>/dev/null; then
        echo -e "${GREEN}✓ SSH key authentication already configured${NC}"
        return 0
    fi

    # Key auth doesn't work, need to set it up
    echo -e "${YELLOW}SSH key not yet authorized on remote server, setting up...${NC}"

    # Use sshpass to copy the SSH key (sshpass is installed by check_prerequisites)
    cat "$SSH_KEY_PATH.pub" | sshpass -p "$SERVER_PASSWORD" ssh -o StrictHostKeyChecking=no "$SERVER_USER@$SERVER_HOST" \
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"

    # Verify key-based auth works now
    sleep 1
    if ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$SERVER_USER@$SERVER_HOST" "exit" 2>/dev/null; then
        echo -e "${GREEN}✓ SSH key authentication configured successfully${NC}"
    else
        echo -e "${RED}Failed to set up SSH key authentication${NC}"
        echo -e "${YELLOW}Please manually copy your SSH key:${NC}"
        echo -e "${BLUE}ssh-copy-id $SERVER_USER@$SERVER_HOST${NC}"
        exit 1
    fi
}

# Execute command on remote server
remote_exec() {
    ssh -o StrictHostKeyChecking=no "$SERVER_USER@$SERVER_HOST" "$@"
}

# Copy files to remote server
remote_copy() {
    rsync -avz --progress -e "ssh -o StrictHostKeyChecking=no" "$@"
}

# Check if this is an initial deployment or update
check_deployment_status() {
    echo -e "${BLUE}Checking deployment status...${NC}"

    if remote_exec "test -d $REMOTE_DEPLOY_DIR"; then
        DEPLOYMENT_MODE="update"
        echo -e "${YELLOW}Existing installation detected - Update mode${NC}"
    else
        DEPLOYMENT_MODE="initial"
        echo -e "${GREEN}No existing installation - Initial deployment mode${NC}"
    fi
}

# Create backup of existing deployment
create_backup() {
    echo -e "${BLUE}Creating backup of existing deployment...${NC}"

    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_dir="${REMOTE_DEPLOY_DIR}_backup_${timestamp}"

    remote_exec "sudo cp -r $REMOTE_DEPLOY_DIR $backup_dir"
    echo -e "${GREEN}✓ Backup created at $backup_dir${NC}"
}

# Stop validator service
stop_service() {
    echo -e "${BLUE}Stopping validator service...${NC}"

    if remote_exec "sudo systemctl is-active --quiet aurelius-validator"; then
        remote_exec "sudo systemctl stop aurelius-validator"
        echo -e "${GREEN}✓ Service stopped${NC}"
    else
        echo -e "${YELLOW}Service is not running${NC}"
    fi
}

# Install system dependencies
install_system_dependencies() {
    echo -e "${BLUE}Installing system dependencies...${NC}"

    remote_exec "sudo apt-get update"
    remote_exec "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-dev \
        python3-pip \
        git \
        build-essential \
        curl \
        wget"

    echo -e "${GREEN}✓ System dependencies installed${NC}"
}

# Configure firewall
configure_firewall() {
    echo -e "${BLUE}Configuring firewall...${NC}"

    remote_exec "sudo ufw --force enable"
    remote_exec "sudo ufw allow 22/tcp"     # SSH
    remote_exec "sudo ufw allow 8091/tcp"   # Validator port
    remote_exec "sudo ufw status"

    echo -e "${GREEN}✓ Firewall configured${NC}"
}

# Create deployment directory
create_directories() {
    echo -e "${BLUE}Creating deployment directories...${NC}"

    remote_exec "sudo mkdir -p $REMOTE_DEPLOY_DIR"
    remote_exec "sudo chown $SERVER_USER:$SERVER_USER $REMOTE_DEPLOY_DIR"
    remote_exec "sudo mkdir -p $REMOTE_DATA_DIR/datasets"
    remote_exec "sudo chown -R $SERVER_USER:$SERVER_USER $REMOTE_DATA_DIR"
    remote_exec "chmod 700 $REMOTE_DATA_DIR"

    echo -e "${GREEN}✓ Directories created${NC}"
}

# Transfer validator files
transfer_files() {
    echo -e "${BLUE}Transferring validator files...${NC}"

    # Create exclude file for rsync
    local exclude_file=$(mktemp)
    cat > "$exclude_file" << EOF
.git/
.venv/
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg-info/
dist/
build/
logs/
datasets/
.env
.pytest_cache/
.coverage
*.log
EOF

    # Transfer main project files
    remote_copy --exclude-from="$exclude_file" \
        "$VALIDATOR_SOURCE_DIR/" \
        "$SERVER_USER@$SERVER_HOST:$REMOTE_DEPLOY_DIR/"

    rm "$exclude_file"

    echo -e "${GREEN}✓ Files transferred${NC}"
}

# Transfer Bittensor wallet
transfer_wallet() {
    echo -e "${BLUE}Transferring Bittensor wallet...${NC}"

    if [[ ! -d "$WALLET_SOURCE_DIR" ]]; then
        echo -e "${YELLOW}Warning: Wallet directory not found at $WALLET_SOURCE_DIR${NC}"
        echo -e "${YELLOW}Please manually create/transfer wallet to remote server${NC}"
        return
    fi

    # Create remote .bittensor directory
    remote_exec "mkdir -p ~/.bittensor"

    # Transfer wallet files
    remote_copy -r "$WALLET_SOURCE_DIR/" \
        "$SERVER_USER@$SERVER_HOST:~/.bittensor/"

    # Set proper permissions
    remote_exec "chmod 700 ~/.bittensor"
    remote_exec "find ~/.bittensor -type f -name 'coldkey' -exec chmod 600 {} \;"
    remote_exec "find ~/.bittensor -type f -name 'hotkey' -exec chmod 600 {} \;"

    echo -e "${GREEN}✓ Wallet transferred${NC}"
}

# Create Python virtual environment
create_virtualenv() {
    echo -e "${BLUE}Creating Python ${PYTHON_VERSION} virtual environment...${NC}"

    remote_exec "cd $REMOTE_DEPLOY_DIR && python${PYTHON_VERSION} -m venv .venv"

    echo -e "${GREEN}✓ Virtual environment created${NC}"
}

# Install Python dependencies
install_dependencies() {
    echo -e "${BLUE}Installing Python dependencies...${NC}"

    remote_exec "cd $REMOTE_DEPLOY_DIR && \
        source .venv/bin/activate && \
        pip install --upgrade pip && \
        pip install -e ."

    echo -e "${GREEN}✓ Dependencies installed${NC}"
}

# Configure environment variables
configure_env() {
    echo -e "${BLUE}Configuring environment...${NC}"

    # Check if .env already exists (update mode)
    if remote_exec "test -f $REMOTE_DEPLOY_DIR/.env"; then
        echo -e "${YELLOW}Existing .env file found - preserving configuration${NC}"
        return
    fi

    # Create .env from testnet template
    remote_exec "cd $REMOTE_DEPLOY_DIR && cp .env.testnet .env"

    # Update paths to absolute paths for production
    remote_exec "cd $REMOTE_DEPLOY_DIR && sed -i 's|DATASET_DIR=.*|DATASET_DIR=$REMOTE_DATA_DIR/datasets|g' .env"
    remote_exec "cd $REMOTE_DEPLOY_DIR && sed -i 's|VALIDATOR_HOST=.*|VALIDATOR_HOST=0.0.0.0|g' .env"
    remote_exec "cd $REMOTE_DEPLOY_DIR && sed -i 's|AUTO_DETECT_EXTERNAL_IP=.*|AUTO_DETECT_EXTERNAL_IP=true|g' .env"

    echo -e "${GREEN}✓ Environment configured${NC}"
    echo -e "${YELLOW}⚠ IMPORTANT: Edit .env on remote server to set OPENAI_API_KEY and wallet names${NC}"
}

# Create systemd service
create_systemd_service() {
    echo -e "${BLUE}Creating systemd service...${NC}"

    # Create service file content
    remote_exec "sudo tee /etc/systemd/system/aurelius-validator.service > /dev/null << 'EOF'
[Unit]
Description=Aurelius Subnet Validator
After=network.target

[Service]
Type=simple
User=$SERVER_USER
WorkingDirectory=$REMOTE_DEPLOY_DIR
Environment=\"PATH=$REMOTE_DEPLOY_DIR/.venv/bin\"
ExecStart=$REMOTE_DEPLOY_DIR/.venv/bin/aurelius-validator
Restart=always
RestartSec=10
StandardOutput=append:$REMOTE_DATA_DIR/validator.log
StandardError=append:$REMOTE_DATA_DIR/validator.error.log

[Install]
WantedBy=multi-user.target
EOF"

    # Reload systemd and enable service
    remote_exec "sudo systemctl daemon-reload"
    remote_exec "sudo systemctl enable aurelius-validator"

    echo -e "${GREEN}✓ Systemd service created${NC}"
}

# Start validator service
start_service() {
    echo -e "${BLUE}Starting validator service...${NC}"

    remote_exec "sudo systemctl start aurelius-validator"
    sleep 3  # Wait for service to start

    echo -e "${GREEN}✓ Service started${NC}"
}

# Verify deployment
verify_deployment() {
    echo -e "${BLUE}Verifying deployment...${NC}"

    # Check service status
    if remote_exec "sudo systemctl is-active --quiet aurelius-validator"; then
        echo -e "${GREEN}✓ Service is active${NC}"
    else
        echo -e "${RED}✗ Service is not active${NC}"
        echo -e "${YELLOW}Checking logs:${NC}"
        remote_exec "sudo journalctl -u aurelius-validator -n 50 --no-pager"
        return 1
    fi

    # Show recent logs
    echo -e "\n${BLUE}Recent logs:${NC}"
    remote_exec "sudo journalctl -u aurelius-validator -n 20 --no-pager"
}

# Display post-deployment instructions
show_instructions() {
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}  Deployment Complete!${NC}"
    echo -e "${GREEN}========================================${NC}\n"

    echo -e "${YELLOW}Next steps:${NC}"
    echo -e "1. SSH to server: ${BLUE}ssh $SERVER_USER@$SERVER_HOST${NC}"
    echo -e "2. Edit configuration: ${BLUE}nano $REMOTE_DEPLOY_DIR/.env${NC}"
    echo -e "   - Set ${YELLOW}OPENAI_API_KEY${NC}"
    echo -e "   - Set ${YELLOW}VALIDATOR_WALLET_NAME${NC} and ${YELLOW}VALIDATOR_HOTKEY${NC}"
    echo -e "3. Restart service: ${BLUE}sudo systemctl restart aurelius-validator${NC}"
    echo -e "4. Check logs: ${BLUE}sudo journalctl -u aurelius-validator -f${NC}"
    echo -e "\n${YELLOW}Service management commands:${NC}"
    echo -e "  Status:  ${BLUE}sudo systemctl status aurelius-validator${NC}"
    echo -e "  Stop:    ${BLUE}sudo systemctl stop aurelius-validator${NC}"
    echo -e "  Start:   ${BLUE}sudo systemctl start aurelius-validator${NC}"
    echo -e "  Restart: ${BLUE}sudo systemctl restart aurelius-validator${NC}"
    echo -e "  Logs:    ${BLUE}sudo journalctl -u aurelius-validator -f${NC}"
    echo -e "\n${YELLOW}Data locations:${NC}"
    echo -e "  Validator: ${BLUE}$REMOTE_DEPLOY_DIR${NC}"
    echo -e "  Datasets:  ${BLUE}$REMOTE_DATA_DIR/datasets${NC}"
    echo -e "  Logs:      ${BLUE}$REMOTE_DATA_DIR/validator.log${NC}"
}

# Check prerequisites
check_prerequisites() {
    # Check if we'll need sshpass for initial setup
    if [[ ! -f "$SSH_KEY_PATH" ]] || ! ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$SERVER_USER@$SERVER_HOST" "exit" 2>/dev/null; then
        # We'll need sshpass for initial SSH key setup
        if ! command -v sshpass &> /dev/null; then
            echo -e "${YELLOW}Note: sshpass is needed for initial SSH key setup${NC}"
            echo -e "${BLUE}Installing sshpass...${NC}"

            # Try to install
            if sudo apt-get update -qq && sudo apt-get install -y sshpass 2>/dev/null; then
                echo -e "${GREEN}✓ sshpass installed${NC}"
            else
                echo -e "${YELLOW}Please install sshpass and run again:${NC}"
                echo -e "${BLUE}  sudo apt-get install sshpass${NC}"
                echo -e "${YELLOW}(This is only needed once for initial SSH key setup)${NC}"
                return 1
            fi
        fi
    fi
    return 0
}

# Main deployment flow
main() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Aurelius Validator Deployment${NC}"
    echo -e "${GREEN}========================================${NC}\n"

    # Parse credentials
    parse_credentials

    # Check prerequisites (will prompt for sudo if needed)
    if ! check_prerequisites; then
        exit 1
    fi

    # Setup SSH key authentication
    setup_ssh_key

    # Check deployment status
    check_deployment_status

    # Handle update mode
    if [[ "$DEPLOYMENT_MODE" == "update" ]]; then
        create_backup
        stop_service
    fi

    # Initial setup (only on first deployment)
    if [[ "$DEPLOYMENT_MODE" == "initial" ]]; then
        install_system_dependencies
        configure_firewall
    fi

    # Create directories
    create_directories

    # Transfer files
    transfer_files

    # Transfer wallet (only on initial deployment)
    if [[ "$DEPLOYMENT_MODE" == "initial" ]]; then
        transfer_wallet
    fi

    # Setup Python environment
    if [[ "$DEPLOYMENT_MODE" == "initial" ]] || ! remote_exec "test -d $REMOTE_DEPLOY_DIR/.venv"; then
        create_virtualenv
    fi

    # Install/update dependencies
    install_dependencies

    # Configure environment
    configure_env

    # Setup systemd service
    if [[ "$DEPLOYMENT_MODE" == "initial" ]] || ! remote_exec "sudo systemctl list-unit-files | grep -q aurelius-validator"; then
        create_systemd_service
    fi

    # Start service
    start_service

    # Verify
    verify_deployment

    # Show instructions
    show_instructions
}

# Run main function
main "$@"

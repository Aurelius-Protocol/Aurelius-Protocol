#!/bin/bash

# Simple script to change expired password on remote server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse credentials
if [[ ! -f "$SCRIPT_DIR/.passw" ]]; then
    echo -e "${RED}Error: .passw file not found${NC}"
    exit 1
fi

passw_content=$(cat "$SCRIPT_DIR/.passw")
OLD_PASSWORD=$(echo "$passw_content" | grep -oP "(?<=-p ')[^']+")
SERVER_USER=$(echo "$passw_content" | grep -oP "ssh \K[^@]+(?=@)")
SERVER_HOST=$(echo "$passw_content" | grep -oP "@\K[0-9.]+")

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}  Remote Server Password Fix${NC}"
echo -e "${YELLOW}========================================${NC}\n"

echo -e "${YELLOW}The remote server password has expired and needs to be changed.${NC}"
echo -e "${BLUE}Server: $SERVER_USER@$SERVER_HOST${NC}"
echo -e "${BLUE}Current password: $OLD_PASSWORD${NC}\n"

echo -e "${YELLOW}Please follow these steps:${NC}"
echo -e "1. Open a new terminal"
echo -e "2. Run: ${BLUE}ssh $SERVER_USER@$SERVER_HOST${NC}"
echo -e "3. When prompted, enter current password: ${BLUE}$OLD_PASSWORD${NC}"
echo -e "4. Enter the same password again when prompted for 'Current password'"
echo -e "5. Enter a NEW password (twice)"
echo -e "6. Type ${BLUE}exit${NC} to logout\n"

echo -e "${YELLOW}After changing the password, update the .passw file:${NC}"
echo -e "   ${BLUE}echo \"sshpass -p 'YOUR_NEW_PASSWORD' ssh $SERVER_USER@$SERVER_HOST\" > $SCRIPT_DIR/.passw${NC}\n"

echo -e "${YELLOW}Then run ./deploy.sh again${NC}\n"

# Offer to try automated approach if expect is available
if command -v expect &> /dev/null; then
    echo -e "${GREEN}Note: 'expect' is installed. Would you like to try automated password change? (y/n)${NC}"
    read -r response
    if [[ "$response" == "y" ]]; then
        ./change-remote-password.sh
    fi
fi

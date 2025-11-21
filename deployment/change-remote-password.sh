#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse credentials
if [[ ! -f "$SCRIPT_DIR/.passw" ]]; then
    echo -e "${RED}Error: .passw file not found${NC}"
    exit 1
fi

passw_content=$(cat "$SCRIPT_DIR/.passw")
OLD_PASSWORD=$(echo "$passw_content" | grep -oP "(?<=-p ')[^']+")
SERVER_USER=$(echo "$passw_content" | grep -oP "ssh \K[^@]+(?=@)")
SERVER_HOST=$(echo "$passw_content" | grep -oP "@\K[0-9.]+")

echo -e "${BLUE}Remote Server Password Change${NC}"
echo -e "${YELLOW}The remote server requires a password change.${NC}"
echo -e "${YELLOW}Current password: $OLD_PASSWORD${NC}\n"

read -sp "Enter new password for $SERVER_USER@$SERVER_HOST: " NEW_PASSWORD
echo
read -sp "Confirm new password: " NEW_PASSWORD_CONFIRM
echo

if [[ "$NEW_PASSWORD" != "$NEW_PASSWORD_CONFIRM" ]]; then
    echo -e "${RED}Passwords don't match!${NC}"
    exit 1
fi

echo -e "${BLUE}Changing password on remote server...${NC}"

# Create expect script to handle password change
expect_script=$(mktemp)
cat > "$expect_script" << 'EOF'
#!/usr/bin/expect -f
set timeout 30
set old_password [lindex $argv 0]
set new_password [lindex $argv 1]
set user [lindex $argv 2]
set host [lindex $argv 3]

spawn ssh -o StrictHostKeyChecking=no $user@$host

expect {
    "password:" {
        send "$old_password\r"
        exp_continue
    }
    "Current password:" {
        send "$old_password\r"
        exp_continue
    }
    "New password:" {
        send "$new_password\r"
        exp_continue
    }
    "Retype new password:" {
        send "$new_password\r"
        exp_continue
    }
    "password updated successfully" {
        send "exit\r"
    }
    "#" {
        send "exit\r"
    }
    timeout {
        puts "Timeout occurred"
        exit 1
    }
}

expect eof
EOF

chmod +x "$expect_script"

if "$expect_script" "$OLD_PASSWORD" "$NEW_PASSWORD" "$SERVER_USER" "$SERVER_HOST"; then
    rm "$expect_script"
    echo -e "${GREEN}✓ Password changed successfully${NC}"

    # Update .passw file
    echo -e "${BLUE}Updating .passw file...${NC}"
    echo "sshpass -p '$NEW_PASSWORD' ssh $SERVER_USER@$SERVER_HOST" > "$SCRIPT_DIR/.passw"
    chmod 600 "$SCRIPT_DIR/.passw"
    echo -e "${GREEN}✓ .passw file updated${NC}"

    echo -e "${GREEN}You can now run ./deploy.sh${NC}"
else
    rm "$expect_script"
    echo -e "${RED}Failed to change password${NC}"
    exit 1
fi

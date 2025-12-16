#!/bin/bash
set -e

# Ensure data directory exists
mkdir -p /var/lib/aurelius/datasets

# Fix Docker bind mount issue: when mounting a file that doesn't exist on the host,
# Docker creates a directory instead. Detect and fix this.
for file in /var/lib/aurelius/miner_scores.json /var/lib/aurelius/validator_trust.json; do
    if [ -d "$file" ]; then
        # Remove directory created by Docker's bind mount
        rmdir "$file" 2>/dev/null || rm -rf "$file"
    fi
    if [ ! -f "$file" ]; then
        echo "{}" > "$file"
    fi
done

exec "$@"

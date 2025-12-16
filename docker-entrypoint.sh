#!/bin/bash
set -e

# Ensure data directories exist
mkdir -p /var/lib/aurelius/datasets

# Initialize JSON files if they don't exist
for file in /var/lib/aurelius/miner_scores.json /var/lib/aurelius/validator_trust.json; do
    if [ ! -f "$file" ]; then
        echo "{}" > "$file"
    fi
done

exec "$@"

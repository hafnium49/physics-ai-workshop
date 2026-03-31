#!/usr/bin/env bash
# Teardown: Remove all workshop user accounts and the workshop group.
# Requires sudo.
#
# Usage:
#   sudo bash setup/teardown.sh

set -euo pipefail

NUM_USERS=5

echo "=== Physics-AI Workshop: Teardown ==="
echo "This will DELETE users engineer1-${NUM_USERS} and all their home directories."
read -p "Are you sure? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

for i in $(seq 1 $NUM_USERS); do
    USER="engineer$i"
    if id "$USER" &>/dev/null; then
        # Kill any running processes
        pkill -u "$USER" 2>/dev/null || true
        sleep 1
        # Remove user and home directory
        userdel -r "$USER" 2>/dev/null || true
        echo "[OK] Removed $USER"
    else
        echo "[SKIP] $USER does not exist"
    fi
done

# Remove workshop group
groupdel workshop 2>/dev/null || true
echo "[OK] Removed workshop group"

echo ""
echo "=== Teardown complete ==="

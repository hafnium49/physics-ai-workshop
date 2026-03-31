#!/usr/bin/env bash
# Phase 1: Create workshop user accounts and provision each one.
# This is the ONLY script that requires sudo.
#
# Usage:
#   sudo bash setup/01_create_users.sh [REPO_DIR] [PASSWORD]
#
# Example:
#   sudo bash setup/01_create_users.sh /home/admin/physics-ai-workshop <WORKSHOP_PASSWORD>

set -euo pipefail

REPO_DIR="${1:?Usage: sudo bash setup/01_create_users.sh REPO_DIR PASSWORD}"
PASSWORD="${2:?Usage: sudo bash setup/01_create_users.sh REPO_DIR PASSWORD}"
NUM_USERS=5

# Resolve absolute path
REPO_DIR="$(cd "$REPO_DIR" && pwd)"

echo "=== Physics-AI Workshop: User Setup ==="
echo "Repo:     $REPO_DIR"
echo "Users:    engineer1 - engineer${NUM_USERS}"
echo ""

# Create workshop group (ignore if exists)
groupadd -f workshop
echo "[OK] Workshop group created"

for i in $(seq 1 $NUM_USERS); do
    USER="engineer$i"
    echo ""
    echo "--- Setting up $USER ---"

    # Create user if not exists (serial — needs sudo)
    if id "$USER" &>/dev/null; then
        echo "[SKIP] User $USER already exists — password NOT changed"
    else
        useradd -m -s /bin/bash -g workshop "$USER"
        echo "$USER:$PASSWORD" | chpasswd
        echo "[OK] User $USER created"
    fi
done

# Temporarily allow traversal to repo dir (h_fujiwara home is 750 from SSH hardening)
HOME_DIR=$(echo "$REPO_DIR" | sed 's|^\(/home/[^/]*\)/.*|\1|')
ORIG_PERMS=$(stat -c '%a' "$HOME_DIR")
chmod 755 "$HOME_DIR"
echo "[OK] Temporarily opened $HOME_DIR for provisioning"

# Run per-user provisioning in parallel (no sudo needed)
echo ""
echo "--- Provisioning all users in parallel ---"
for i in $(seq 1 $NUM_USERS); do
    USER="engineer$i"
    sudo -u "$USER" bash "$REPO_DIR/setup/02_provision_user.sh" "$REPO_DIR" "$i" &
done
wait
echo "[OK] All users provisioned"

# Restore permissions
chmod "$ORIG_PERMS" "$HOME_DIR"
echo "[OK] Restored $HOME_DIR to $ORIG_PERMS"

echo ""
echo "=== All $NUM_USERS users provisioned ==="
echo ""
echo "Next steps:"
echo "  1. Run Phase 3 (OAuth): sudo su - engineer1 && claude login"
echo "  2. Run pre-flight:      sudo -u engineer1 bash -c 'cd ~/physics_sim && MUJOCO_GL=egl python scripts/preflight.py'"

#!/usr/bin/env bash
# Phase 2: Provision a single workshop user (no sudo required).
# Called by 01_create_users.sh for each user, or manually:
#
# Usage (as the target user):
#   bash setup/02_provision_user.sh REPO_DIR USER_NUMBER
#
# Example:
#   sudo -u engineer1 bash setup/02_provision_user.sh /home/admin/physics-ai-workshop 1

set -euo pipefail

REPO_DIR="${1:?Usage: bash setup/02_provision_user.sh REPO_DIR USER_NUMBER}"
USER_NUM="${2:?Usage: bash setup/02_provision_user.sh REPO_DIR USER_NUMBER}"
STREAM_PORT="1808${USER_NUM}"

echo "  Provisioning user $(whoami) (port $STREAM_PORT)..."

# --- 1. Install nvm (Node Version Manager) ---
export NVM_DIR="$HOME/.nvm"
if [ ! -d "$NVM_DIR" ]; then
    curl -fsSo "/tmp/nvm_install_${USER_NUM}.sh" https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh
    bash "/tmp/nvm_install_${USER_NUM}.sh" >/dev/null 2>&1
    rm -f "/tmp/nvm_install_${USER_NUM}.sh"
    echo "  [OK] nvm installed"
else
    echo "  [SKIP] nvm already installed"
fi

# Load nvm
export NVM_DIR="$HOME/.nvm"
# shellcheck disable=SC1091
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

# --- 2. Install Node.js ---
if command -v node &>/dev/null; then
    echo "  [SKIP] Node.js $(node --version) already installed"
else
    nvm install 20 >/dev/null 2>&1
    nvm use 20 >/dev/null 2>&1
    echo "  [OK] Node.js $(node --version) installed"
fi

# --- 3. Install Claude Code ---
if command -v claude &>/dev/null; then
    echo "  [SKIP] Claude Code $(claude --version 2>/dev/null || echo 'installed') already installed"
else
    npm install -g @anthropic-ai/claude-code >/dev/null 2>&1
    echo "  [OK] Claude Code installed"
fi

# --- 4. Create Python virtual environment (always run pip — it's idempotent) ---
python3 -m venv "$HOME/workshop_env" 2>/dev/null || true
"$HOME/workshop_env/bin/pip" install --quiet mujoco mediapy numpy matplotlib Pillow
if [ $? -eq 0 ]; then
    echo "  [OK] Python venv created with mujoco, mediapy, numpy, matplotlib, Pillow"
fi

# --- 5. Set up .bashrc (idempotent) ---
BASHRC="$HOME/.bashrc"
grep -q "workshop_env" "$BASHRC" 2>/dev/null || echo 'source ~/workshop_env/bin/activate' >> "$BASHRC"
grep -q "STREAM_PORT" "$BASHRC" 2>/dev/null || echo "export STREAM_PORT=$STREAM_PORT" >> "$BASHRC"
grep -q "MUJOCO_GL" "$BASHRC" 2>/dev/null || echo 'export MUJOCO_GL=egl' >> "$BASHRC"
echo "  [OK] .bashrc configured (venv, STREAM_PORT=$STREAM_PORT, MUJOCO_GL=egl)"

# --- 6. Clone workspace from GitHub ---
WORKSPACE="$HOME/physics_sim"
git config --global user.name "engineer${USER_NUM}"
git config --global user.email "engineer${USER_NUM}@workshop.local"

if [ -d "$WORKSPACE/.git" ]; then
    echo "  [SKIP] Workspace already cloned — pulling latest"
    cd "$WORKSPACE" && git pull -q 2>/dev/null || true
else
    rm -rf "$WORKSPACE" 2>/dev/null
    git clone -q https://github.com/hafnium49/physics-ai-workshop.git "$WORKSPACE"
    echo "  [OK] Workspace cloned from GitHub"
fi

echo "  [DONE] User $(whoami) ready"

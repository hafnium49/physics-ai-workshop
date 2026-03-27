# Host Preparation Runbook: FY2026 Physics-AI Workshop Environments

**Target Hardware:** NVIDIA DGX Spark (or compatible)
**Objective:** Provision five fully isolated Python/MuJoCo environments and authenticate a single Claude Max subscription across all five accounts using secure OAuth device authorization.

## 1. Executive Summary

This document outlines the software and environment preparation required on the host machine prior to the 1-hour Physics-AI workshop. The five participating material engineers will connect via VS Code Remote SSH into isolated home directories. They need pre-configured Python virtual environments containing MuJoCo and mediapy. To share a single consumer Claude Max subscription without distributing credentials, the host must manually perform browserless OAuth authentication for each account in advance.

> **Note:** This runbook covers software/environment setup only. Network access and SSH tunnel configuration are handled separately by the host administrator.

---

## 2. Phase 1: Install Global Dependencies

Claude Code operates as a Node.js application. It must be installed globally so all workshop participants can execute it from their respective terminal sessions.

```bash
# 1. Install Node.js (Version 20.x recommended)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# 2. Install Claude Code globally via npm
sudo npm install -g @anthropic-ai/claude-code

# 3. Create the dedicated user group for easy teardown later
sudo groupadd workshop
```

---

## 3. Phase 2: Provision Isolated Workspaces & Python Environments

This automated script creates the five users, sets their temporary passwords, provisions isolated Python virtual environments, and installs the necessary physics simulation libraries.

```bash
for i in {1..5}; do
    echo "Configuring environment for engineer$i..."

    # 1. Create user and set temporary password
    sudo useradd -m -s /bin/bash -g workshop engineer$i
    echo "engineer$i:<WORKSHOP_PASSWORD>" | sudo chpasswd

    # 2. Create Python virtual environment
    sudo -u engineer$i bash -c "cd ~ && python3 -m venv workshop_env"

    # 3. Install MuJoCo and MediaPy (for rendering .mp4s without a GUI)
    sudo -u engineer$i bash -c "~/workshop_env/bin/pip install mujoco mediapy numpy"

    # 4. Auto-activate the environment upon login
    sudo -u engineer$i bash -c "echo 'source ~/workshop_env/bin/activate' >> ~/.bashrc"

    # 5. Create a clean working directory and copy workshop content
    sudo -u engineer$i bash -c "mkdir -p ~/physics_sim"

    # 6. Copy MuJoCo content (Panda model + ball_and_plate) and streamer into workspace
    # Adjust the source path to where you cloned the physics-ai-workshop repo
    sudo cp -r /path/to/physics-ai-workshop/content/* /home/engineer$i/physics_sim/
    sudo cp /path/to/physics-ai-workshop/mujoco_streamer.py /home/engineer$i/physics_sim/

    # 7. Set per-user streaming port to avoid collisions
    echo "export STREAM_PORT=808$i" | sudo tee -a /home/engineer$i/.bashrc > /dev/null

    sudo chown -R engineer$i:workshop /home/engineer$i/physics_sim/
done
```

> **Replace** `<WORKSHOP_PASSWORD>` with the actual password for each user. For better security, generate unique passwords per user (see your SSH setup scripts).

---

## 4. Phase 3: The Claude Code Authentication Trick (OAuth)

Because you are using a consumer Claude Max subscription rather than an Enterprise API key, you cannot simply export an API key variable. You must authenticate each account manually.

By initiating the login from the host terminal but completing the authorization in your personal web browser, you securely drop an active session token into each engineer's profile. They will never see your password.

**Preparation on your personal laptop:**
Ensure you are logged into your Anthropic/Claude account in your primary web browser.

**Execute the following manual steps for ALL 5 accounts on the host:**

1. **Switch to the first engineer's account:**
   ```bash
   sudo su - engineer1
   ```
2. **Initiate the Claude Code login:**
   ```bash
   claude login
   ```
3. **Authorize via your browser:**
   The terminal will output a specific URL (e.g., `https://claude.ai/device?code=XXXX-XXXX`).
   * Copy this URL.
   * Paste it into your personal laptop's web browser.
   * Click **"Authorize"** on the webpage.
   * *Look back at the host terminal: it should now say "Successfully logged in!"*
4. **Exit and repeat:**
   ```bash
   exit  # This returns you to your admin account
   ```
5. **Repeat steps 1-4 for `engineer2` through `engineer5`.**

---

## 5. Phase 4: Pre-Flight Validation

To guarantee a zero-friction experience for the engineers, perform a single end-to-end test on one of the provisioned accounts.

1. From your personal laptop, open VS Code and use the Remote SSH extension to connect as a participant would (using the connection details from your SSH setup).
2. Enter the workshop password.
3. Open a new terminal in VS Code (`Ctrl + ~`).
4. Verify the prompt shows `(workshop_env) engineer1@<hostname>`.
5. Run the command: `claude -p "Say hello to the workshop"`
6. If Claude responds without asking for a login, **your environments are perfectly provisioned.**

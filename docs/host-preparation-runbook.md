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
    sudo -u engineer$i bash -c "~/workshop_env/bin/pip install mujoco mediapy numpy matplotlib Pillow"

    # 4. Auto-activate the environment upon login
    sudo -u engineer$i bash -c "echo 'source ~/workshop_env/bin/activate' >> ~/.bashrc"

    # 5. Create a clean working directory and copy workshop content
    sudo -u engineer$i bash -c "mkdir -p ~/physics_sim"

    # 6. Copy MuJoCo content (Panda model + ball_and_plate) and streamer into workspace
    # Adjust the source path to where you cloned the physics-ai-workshop repo
    sudo cp -r /path/to/physics-ai-workshop/content/* /home/engineer$i/physics_sim/
    sudo cp /path/to/physics-ai-workshop/mujoco_streamer.py /home/engineer$i/physics_sim/
    sudo cp -r /path/to/physics-ai-workshop/scripts /home/engineer$i/physics_sim/scripts/
    sudo cp /path/to/physics-ai-workshop/.gitignore /home/engineer$i/physics_sim/

    # 7. Set per-user streaming port to avoid collisions
    echo "export STREAM_PORT=1808$i" | sudo tee -a /home/engineer$i/.bashrc > /dev/null
    echo "export MUJOCO_GL=egl" | sudo tee -a /home/engineer$i/.bashrc > /dev/null

    # 8. Copy CLAUDE.md, Claude Code settings, and skills into workspace
    sudo cp /path/to/physics-ai-workshop/CLAUDE.md /home/engineer$i/physics_sim/
    sudo mkdir -p /home/engineer$i/physics_sim/.claude
    sudo cp /path/to/physics-ai-workshop/.claude/settings.json /home/engineer$i/physics_sim/.claude/
    sudo cp -r /path/to/physics-ai-workshop/.claude/skills /home/engineer$i/physics_sim/.claude/skills/

    # 9. Fix ownership before git init (git init must run as the user)
    sudo chown -R engineer$i:workshop /home/engineer$i/physics_sim/

    # 10. Initialize git repo (Claude Code resolves settings from git root)
    sudo -u engineer$i bash -c "cd ~/physics_sim && git init && git add -A && git commit -m 'Workshop setup'"
done
```

> **Replace** `<WORKSHOP_PASSWORD>` with the actual password for each user. For better security, generate unique passwords per user (see your SSH setup scripts).

> **Note:** The `scripts/` directory contains reference scripts with spoiler comments. These are intentionally provided so participants can run them and focus on the exploration sprint.

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

### Automated pre-flight check

From the cloned repo directory on the host:

```bash
cd /path/to/physics-ai-workshop
MUJOCO_GL=egl python scripts/preflight.py
```

All 9 checks should print `[PASS]`. If any fail, fix the issue before proceeding.

### Manual end-to-end test

1. From your personal laptop, open VS Code and use the Remote SSH extension to connect as `engineer1`.
2. Enter the workshop password.
3. Open a new terminal in VS Code (`Ctrl + ~`).
4. Verify the prompt shows `(workshop_env) engineer1@<hostname>`.
5. Run: `claude -p "Load panda_ball_balance.xml, set the arm to the home pose, place the ball on the plate, and start a live stream using mujoco_streamer.py"`
6. Verify VS Code shows the port forwarding notification — click it and confirm you can see the live video in your browser.
7. If Claude responds without asking for a login and the stream is visible, **your environments are perfectly provisioned.**

---

## 6. Reference Scripts & Sprint Mapping

These are pre-built reference scripts that are pre-copied into participant workspaces. Participants run these directly and use Claude Code to explore and improve them.

| Script | Sprint | Purpose | Example |
|--------|--------|---------|---------|
| `preflight.py` | Pre-workshop (host only) | Validates model, streamer, PID, rendering | `MUJOCO_GL=egl python scripts/preflight.py` |
| `01_validate_assembly.py` | Sprint 1: Explore | Load model, stream live, see robot + ball | `python scripts/01_validate_assembly.py` |
| `02_pid_baseline.py` | Sprint 2: PID Discovery | Deliberately wrong PID baseline (~0.3s survival) | `python scripts/02_pid_baseline.py --kp 50 --kd 10` |
| `03_optimize_pid.py` | Sprint 2: PID Discovery | Grid search proving correct joints exist (host validation) | `python scripts/03_optimize_pid.py --no-render` |
| `04_survival_map.py` | Sprint 3+4: Baseline & Exploration | Survival map with Controller Score metric for controller comparison | `python scripts/04_survival_map.py --kp 2 --kd 0` |
| `05_challenge.py` | Sprint 3+4: First Iteration & Exploration | Controller exploration playground (participants edit via Claude) | `python scripts/04_survival_map.py --controller scripts/05_challenge.py` |

> All scripts support `--no-stream` (saves .mp4 fallback) and `--port` (default: `STREAM_PORT` env var or 18080). Scripts resolve model paths relative to their own location, so they work from any working directory.

> **Leaderboard tip:** Write participant Controller Scores on the whiteboard during Sprint 4. Show multiple runs per participant to celebrate improvement trajectories, not just the highest score.

### "Break glass" fallback

If a participant's Claude Code session is stuck at Sprint 2 after 25 minutes and cannot discover the correct joints, the host can quietly offer this hint:

> *"Try telling Claude: Focus on joint 6 and joint 7 for the PID control, and make sure the correction sign is positive. Use small gains like Kp=2."*

If Sprint 2 is completely blocked, copy `05_challenge.py` into the participant's workspace as a working reference to unblock Sprint 3:

```bash
sudo cp /path/to/physics-ai-workshop/scripts/05_challenge.py /home/engineer$i/physics_sim/
sudo chown engineer$i:workshop /home/engineer$i/physics_sim/05_challenge.py
```

---

## 7. Workshop Day: Running the Autonomous Demo

Before participants start their own sessions, run the interactive demo to set the stage. Follow the step-by-step guide in `docs/autonomous-demo-script.md`.

**Quick summary:**
1. Open Claude Code on the host account (or any engineer account)
2. Show only the browser stream on the projector — not the terminal
3. Paste 5 prompts in sequence: load model → first PID → diagnose joints → fix → add disturbances
4. Total time: 3-5 minutes
5. Then tell participants: *"Now it's your turn."*

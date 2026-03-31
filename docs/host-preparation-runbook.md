# Host Preparation Runbook: FY2026 Physics-AI Workshop Environments

**Target Hardware:** NVIDIA DGX Spark (or compatible)
**Objective:** Provision five fully isolated Python/MuJoCo environments and authenticate a single Claude Max subscription across all five accounts using secure OAuth device authorization.

## 1. Executive Summary

This document outlines the software and environment preparation required on the host machine prior to the 1-hour Physics-AI workshop. The five participating material engineers will connect via VS Code Remote SSH into isolated home directories. They need pre-configured Python virtual environments containing MuJoCo and mediapy. To share a single consumer Claude Max subscription without distributing credentials, the host must manually perform browserless OAuth authentication for each account in advance.

> **Note:** This runbook covers software/environment setup only. Network access and SSH tunnel configuration are handled separately (see Phase 0 below).

---

## 2. Phase 0: Network & SSH Tunnel Setup

The SSH tunnel architecture (GX10 → OCI VPS → Azure VM → participant laptops) is managed by the `dgx-spark-playbooks` repository. Complete these steps BEFORE running Phase 1 below.

See:
- `dgx-spark-playbooks/docs/workshop-ssh-softening-plan.md` — full architecture, security hardening, dead-man's switch
- `dgx-spark-playbooks/bootstrap-workshop.sh` — automated GX10 tunnel + SSH config setup
- `dgx-spark-playbooks/smoke-test-workshop.sh` — tunnel and port verification

Key items that must be in place:
- SSH password auth scoped to `engineer1-5` via `/etc/ssh/sshd_config.d/50-workshop-password-auth.conf`
- GX10 reverse tunnel to OCI VPS running (port 22222)
- Azure VM gateway tunnel running
- `/home/h_fujiwara` permissions set to 750

---

## 3. Phase 1: Create Users & Provision Environments

Everything is handled by two setup scripts in the `setup/` directory. Each user gets a fully isolated local environment — no global Node.js or system-level Python changes.

**Per-user architecture:**
```
/home/engineer1/
├── .nvm/                  # Local Node.js (via nvm)
├── .local/bin/claude      # Local Claude Code
├── workshop_env/          # Local Python venv (mujoco, mediapy, numpy)
└── physics_sim/           # Workspace (content, scripts, CLAUDE.md, .claude/)
```

### Run the setup

```bash
cd /path/to/physics-ai-workshop
sudo bash setup/01_create_users.sh "$(pwd)" "<WORKSHOP_PASSWORD>"
```

This single command:
1. Creates `engineer1` through `engineer5` with the given password
2. For each user, runs `setup/02_provision_user.sh` which:
   - Installs `nvm` + Node.js 20 (per-user, no sudo)
   - Installs Claude Code via npm (per-user)
   - Creates Python venv with `mujoco`, `mediapy`, `numpy`, `matplotlib`, `Pillow`
   - Sets `.bashrc`: auto-activate venv, `STREAM_PORT=1808N`, `MUJOCO_GL=egl`
   - Copies workspace files (content, scripts, streamer, CLAUDE.md, .claude/)
   - Initializes git repo (Claude Code needs this to find settings)

All steps are idempotent — safe to re-run if something fails partway through.

### Teardown after the workshop

```bash
sudo bash setup/teardown.sh
```

Removes all 5 user accounts, home directories, and the workshop group.

> **Note:** The `scripts/` directory is intentionally copied to participant workspaces. Participants run these scripts directly during Sprints 1-3 and use Claude Code to explore and improve controllers in Sprints 4-5.

---

## 4. Phase 2: The Claude Code Authentication Trick (OAuth)

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

## 5. Phase 3: Pre-Flight Validation

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
| `03_optimize_pid.py` | Sprint 3: Working Controller | Working PID with correct joints — participants observe and compare with baseline | `python scripts/03_optimize_pid.py` |
| `04_survival_map.py` | Sprint 4+5: Experiment & R&D | Survival map with Controller Score metric for controller comparison | `python scripts/04_survival_map.py --kp 2 --kd 0` |
| `05_challenge.py` | Sprint 4+5: Experiment & R&D | Controller exploration playground (participants edit via Claude) | `python scripts/04_survival_map.py --controller scripts/05_challenge.py` |

> All scripts support `--no-stream` (saves .mp4 fallback) and `--port` (default: `STREAM_PORT` env var or 18080). Scripts resolve model paths relative to their own location, so they work from any working directory.

> **Leaderboard tip:** Write participant Controller Scores on the whiteboard during Sprints 4 and 5. Show multiple runs per participant to celebrate improvement trajectories, not just the highest score.

### "Break glass" fallback

If a participant's Claude Code session is stuck at Sprint 2 after 25 minutes and cannot discover the correct joints, the host can quietly offer this hint:

> *"Try telling Claude: Focus on joint 6 and joint 7 for the PID control, and make sure the correction sign is positive. Use small gains like Kp=2."*

If Sprint 2 is completely blocked, copy `05_challenge.py` into the participant's workspace as a working reference to unblock Sprint 4:

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

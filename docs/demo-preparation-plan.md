# Workshop Demo Preparation Plan

## Context

Build and validate the full demo pipeline for the Physics-AI workshop. Material engineers (non-programmers) use Claude Code to control a Franka Panda arm balancing a ball on a plate via PID control.

**Key pivot:** Real-time MJPEG web streaming replaces .mp4 output. VS Code Remote SSH auto-forwards the port — engineers see live simulation in browser. Iteration time drops from ~90s to ~15s.

**Key physics finding:** With position actuators, PID gain tuning is trivially easy once the correct joints (j5+j6) and sign (+) are found. Static balancing alone won't fill 30 minutes. **Solution: progressive disturbance challenges** restore the difficulty curve.

**Robustness hardening:** All scripts resolve model paths relative to script location (not cwd), read streaming port from `STREAM_PORT` env var (not hardcoded 18080), handle port conflicts with friendly error messages, and recover from NaN without killing the stream.

---

## Step 1: Create the combined world XML — DONE

**File:** `content/panda_ball_balance.xml` (created, verified: 14 bodies, 10 joints, 8 actuators)

- Plate rigidly nested inside `<body name="hand">`, ball as top-level free body
- Scripts reposition ball onto plate after `mj_forward`
- Without PID: ball off in ~1s. With correct PID (Kp=2, joint6+joint7): hits 10s.

---

## Step 2: Create the MJPEG live streamer helper — DONE

**File:** `mujoco_streamer.py` (created, verified: starts/stops cleanly)

3-line API for Claude:
```python
from mujoco_streamer import LiveStreamer
streamer = LiveStreamer()
streamer.start()
# In sim loop: streamer.update(renderer.render())
```

Implementation details:
- **Pillow for JPEG encoding** (NOT opencv-python) — already a transitive dep of mediapy, zero new installs
- `http.server` + `socketserver.ThreadingMixIn` — simplest correct approach
- **Single-slot frame buffer** with `threading.Condition` — constant memory, latest frame only
- `Condition.wait(timeout=1.0)` in handlers for clean Ctrl+C shutdown
- `daemon_threads = True` on server
- HTML page at `/` with auto-reconnect JS (handles backgrounded tabs)
- FPS counter + connection status dot in the HTML overlay
- Suppress HTTP access logging for clean terminal
- Bind `0.0.0.0` for VS Code port detection
- **Per-user ports:** default `port=18080`, but host runbook assigns 18081-18085 per user to avoid collision on shared machine
- `.stop()` method: sets `_running=False`, wakes all waiters, calls `server.shutdown()`
- Port conflict: `start()` catches `OSError` and prints friendly "Port N already in use" message instead of raw traceback
- Docstring warns: render on sim thread only (OpenGL context is not thread-safe)

**All scripts get `--no-stream` fallback:** try/except import, falls back to mediapy .mp4 output.

---

## Step 3: Update assembly script with live stream (Sprint 1) — DONE

**File:** `scripts/01_validate_assembly.py` (updated, verified)

- Stream live via `LiveStreamer` (with .mp4 fallback)
- `while True` loop with Ctrl+C to stop
- Print diagnostics every second
- **New: joint exploration mode** — accept keyboard/terminal commands to nudge individual joints, so participants build intuition for which joints control what

---

## Step 4: Update baseline PID script (Sprint 2) — DONE

**File:** `scripts/02_pid_baseline.py` (updated, verified: 0.8s survival with wrong baseline)

- Add `LiveStreamer` integration
- `while True` loop with auto-reset on ball fall (reposition ball, restart timer)
- Keep `Survival Time: X.X seconds` terminal output (Claude reads this)
- **Add per-joint authority diagnostics** on each reset: "joint7 correction = 0.05 rad → plate tilt change = 0.001 rad" — gives Claude data to reason from, prevents gain-tuning dead-end spiral
- Deliberate baseline: correct joints (j6+j7) but wrong sign → ~0.3s survival
- Ball fall animation: after ball leaves plate, simulation continues until ball hits floor (~0.4s fall + 0.5s settle) before auto-reset

---

## Step 5: Create disturbance challenge script (Sprint 3 content) — DONE

**File:** `scripts/04_challenge.py` (created, syntax verified)

Progressive difficulty that restores the tuning arc:

1. **Level 1 — Static hold** (trivial with correct joints): ball stays for 10s. Confirms working PID.
2. **Level 2 — Periodic impulses**: random force pushes on ball every 2s. Gains matter now. Target: survive 10s.
3. **Level 3 — Moving target**: plate tilts in slow sinusoidal circle while keeping ball centered. Target: survive at increasing speeds.
4. **Level 4 — Speed record**: fastest circle speed where ball survives 10s. Competitive target for leaderboard.

Each level visually distinct on the live stream. Terminal prints current level, survival time, max perturbation survived.

---

## Step 6: Run optimization validation — DONE

**File:** `scripts/03_optimize_pid.py` (updated, validated)

**Critical bug found and fixed:** Axis mapping was swapped — `(jx=4, jy=5)` mapped joint5→X, joint6→Y which is backwards. Correct mapping: `(jx=5, jy=4)` = joint6(ctrl[5])→X, joint5(ctrl[4])→Y.

Validation results (dry run, no video):
- Phase 1: `j6(X)+j7(Y) sign=+1` → 10.0s (WORKS). Wrong signs fail at ~1.2s.
- Phase 2: Correct pairing with Kp=2, Kd=0 hits 10.0s

---

## Step 7: Update workshop materials — DONE

**`CLAUDE.md`** — updated:
- Streaming as primary output, .mp4 as fallback
- `panda_ball_balance.xml` documented
- Physics-only hint (no answer reveal)
- `MUJOCO_GL=egl` requirement
- Ball repositioning code reference

**`docs/participant-guide.md`** — updated:
- Sprint 1: load pre-built model + joint exploration
- Sprint 2: PID discovery (human-AI collaboration)
- Sprint 3: progressive disturbance challenges
- Tips and troubleshooting updated for live streaming

**`docs/host-preparation-runbook.md`** — updated:
- Copy `mujoco_streamer.py` to workspaces
- Per-user ports (18081-18085)
- Pre-flight test with live streaming

---

## Key technical decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Plate attachment | Rigid kinematic (child of `hand`) | DONE; eliminates weld wobble |
| Ball placement | Top-level free body, repositioned in script | MuJoCo free joint constraint |
| PID joints | joint6 (X) + joint7 (Y) | Empirically verified at rotated-plate pose |
| PID sign | Positive (+) | Empirically verified |
| PID gains | Kp=2, Kd=0 | Low gains sufficient due to plate geometry |
| Home pose (wrist) | j5=1.184, j6=3.184, j7=1.158 | Plate horizontal with edge gripped by fingers |
| Visualization | MJPEG stream via Pillow + http.server | Zero new deps; VS Code auto-forwards |
| JPEG encoding | Pillow (not opencv) | Already installed via mediapy |
| Port allocation | 18081-18085 per user | Avoid collision on shared machine |
| Sprint 3 content | Progressive disturbances | Restores difficulty curve; gains matter under perturbation |
| CLAUDE.md hints | Physics-only (not full answer) | Claude converges in 3-5 iterations |
| Fallback | All scripts: --no-stream → .mp4 via mediapy | Insurance if streaming breaks |

---

## Execution progress

1. ~~Create `mujoco_streamer.py`~~ DONE
2. ~~Update `CLAUDE.md`~~ DONE
3. ~~Update scripts 01, 02 with streaming + fallback~~ DONE (+ `send_frame` → `update` API fix)
4. ~~Create `scripts/04_challenge.py`~~ DONE
5. ~~Run `scripts/03_optimize_pid.py`~~ DONE (axis mapping bug fixed)
6. ~~Update participant-guide.md and host-runbook.md~~ DONE
7. ~~Pre-flight sanity check~~ DONE
8. ~~Workshop agent system~~ DONE (simplified from 3-review consensus)
9. ~~Port migration 8080→18080~~ DONE (all 11 files, preflight passes)
10. ~~Robustness hardening~~ DONE (model paths, port env var, NaN recovery, auto-reset loops, port conflict error)

---

## Step 8: Pre-flight sanity check — DONE

**File:** `scripts/preflight.py` (created, all 9 checks pass)

Bugs found and fixed during implementation:
- `LiveStreamer(renderer, port=...)` in scripts 02 and 04 passed extra `renderer` arg — removed
- `_reset_scene` and joint authority check used raw qpos indices instead of `jnt_qposadr` — fixed (free joint shifts qpos layout)
- Streamer import failed from `scripts/` — added project root to `sys.path` in preflight.py
- Same `sys.path` fix applied to all 4 scripts (01, 02, 03, 04) — streamer now imports correctly from any working directory

Results:
```
[PASS] MUJOCO_GL=egl
[PASS] Model loads (14 bodies, 10 joints, 8 actuators)
[PASS] Ball positioning (z_gap=0.0250)
[PASS] Streamer lifecycle (start/stop)
[PASS] mediapy import
[PASS] Correct PID survival (10.0s)
[PASS] Wrong PID fails (0.8s < 2.0s)
[PASS] Joint authority (j5=0.0004, j6=0.0032, j7=0.0000)
[PASS] EGL rendering (480x640x3)
ALL CHECKS PASSED
```

Confirmed: CLAUDE.md does NOT reveal correct joint pairing, sign, or gains (physics-only hints).

---

## Verification

1. ~~`scripts/preflight.py`~~ ALL 9 CHECKS PASSED
2. ~~`01_validate_assembly.py` in stream mode~~ Streamer starts, prints "MuJoCo streamer running" (no .mp4 fallback)
3. Manual: fresh Claude Code session with updated CLAUDE.md converges on correct joints in ≤5 iterations

---

## Step 9: Workshop agent system — DONE

Three reviews (Software Architect, Game Designer, Reality Checker) converged: original plan was over-engineered. Simplified to 3 items:

### Built:

1. **`.claude/settings.json`** — clean permission patterns replacing 43-line ad-hoc allowlist. Covers `python`, `python3`, `MUJOCO_GL=egl python`, `git`, `kill`, `ls`, etc. Participants won't see permission prompts.

2. **CLAUDE.md behavioral nudge** — one line: "When writing a PID controller for the first time, do not run a systematic joint authority analysis upfront." Prevents Claude from being too clever without scripting deliberate failure.

3. **`docs/autonomous-demo-script.md`** — interactive host guide (NOT `claude -p` mega-prompt). 5 prompts the host pastes in sequence: load → first PID → diagnose → fix → challenge. Show only browser stream on projector.

### Critical fixes applied to host runbook:
- `git init` in participant workspaces (Claude Code resolves settings from git root)
- Copy `CLAUDE.md` and `.claude/settings.json` to workspaces
- `MUJOCO_GL=egl` in participant `.bashrc`
- Do NOT copy reference scripts (spoiler comments visible to Claude)

### Killed (from reviews):
- Custom slash commands — feature doesn't exist as user-defined files in Claude Code
- Scripted failure behavior — patronizing; natural failure is more educational
- Facilitator dashboard — overkill for 5 participants
- `claude -p` mega-prompt — single-turn can't produce multi-turn discovery narrative
- Hooks — env vars via `.bashrc` and settings.json are simpler

### Still needs testing:
- `env` block in settings.json — unknown if it propagates to Bash subprocesses
- Permission pattern matching — verify `Bash(python:*)` works empirically
- Full dry run as engineer1 with `git init` workspace

---

## Step 10: Port migration 8080→18080 — DONE

Port 8080 is commonly used by web servers/proxies and likely occupied on the DGX Spark. Migrated all ports to the 18080 range.

| User | Port |
|------|------|
| Host demo | 18080 |
| engineer1 | 18081 |
| engineer2-5 | 18082-18085 |

Changes across 11 files:
- `mujoco_streamer.py` — `__init__` now auto-reads `STREAM_PORT` env var, falls back to 18080. CLAUDE.md examples simplified to `LiveStreamer()` (no port arg).
- All 4 reference scripts — argparse default `8080` → `18080`
- All 5 docs — port references updated
- `.claude/settings.local.json` — removed stale `--port 8080` test permission

Verified: `grep -rn "8080" *.py *.md` returns only `18080` matches. Preflight all 9 checks pass.

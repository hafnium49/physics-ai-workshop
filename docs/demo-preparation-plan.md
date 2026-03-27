# Workshop Demo Preparation Plan

## Context

Build and validate the full demo pipeline for the Physics-AI workshop. Material engineers (non-programmers) use Claude Code to control a Franka Panda arm balancing a ball on a plate via PID control.

**Key pivot:** Real-time MJPEG web streaming replaces .mp4 output. VS Code Remote SSH auto-forwards the port — engineers see live simulation in browser. Iteration time drops from ~90s to ~15s.

**Key physics finding:** With position actuators, PID gain tuning is trivially easy once the correct joints (j5+j6) and sign (+) are found. Static balancing alone won't fill 30 minutes. **Solution: progressive disturbance challenges** restore the difficulty curve.

---

## Step 1: Create the combined world XML — DONE

**File:** `content/panda_ball_balance.xml` (created, verified: 14 bodies, 10 joints, 8 actuators)

- Plate rigidly nested inside `<body name="hand">`, ball as top-level free body
- Scripts reposition ball onto plate after `mj_forward`
- Without PID: ball off in ~1.3s. With correct PID: any Kp/Kd hits 10s.

---

## Step 2: Create the MJPEG live streamer helper — DONE

**File:** `mujoco_streamer.py` (created, verified: starts/stops cleanly)

3-line API for Claude:
```python
from mujoco_streamer import LiveStreamer
streamer = LiveStreamer(port=8080)
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
- **Per-user ports:** default `port=8080`, but host runbook assigns 8081-8085 per user to avoid collision on shared machine
- `.stop()` method: sets `_running=False`, wakes all waiters, calls `server.shutdown()`
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
- Deliberate baseline: wrong joints (j6+j7) + wrong sign → 0.8s survival

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
- Phase 1: `j6(X)+j5(Y) sign=+1` → 5.0s (WORKS). Swapped axes and wrong signs fail.
- Phase 1: `j6(X)+j7(Y) sign=+1` → 5.0s (also works — j7 has no Y authority but Y drift is minimal)
- Phase 2: ALL 20 Kp/Kd combos with correct pairing hit 10.0s

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
- Per-user ports (8081-8085)
- Pre-flight test with live streaming

---

## Key technical decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Plate attachment | Rigid kinematic (child of `hand`) | DONE; eliminates weld wobble |
| Ball placement | Top-level free body, repositioned in script | MuJoCo free joint constraint |
| PID joints | joint5 (Y) + joint6 (X) | Empirically verified; j7 ≈ zero authority |
| PID sign | Positive (+) | Empirically verified |
| Joint6 home | 1.8 rad (mid-range) | Asymmetric range [-0.0175, 3.7525] |
| Visualization | MJPEG stream via Pillow + http.server | Zero new deps; VS Code auto-forwards |
| JPEG encoding | Pillow (not opencv) | Already installed via mediapy |
| Port allocation | 8081-8085 per user | Avoid collision on shared machine |
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

---

## Step 8: Pre-flight sanity check — DONE

**File:** `scripts/preflight.py` (created, all 9 checks pass)

Bugs found and fixed during implementation:
- `LiveStreamer(renderer, port=...)` in scripts 02 and 04 passed extra `renderer` arg — removed
- `_reset_scene` and joint authority check used raw qpos indices instead of `jnt_qposadr` — fixed (free joint shifts qpos layout)
- Streamer import failed from `scripts/` — added project root to `sys.path`

Results:
```
[PASS] MUJOCO_GL=egl
[PASS] Model loads (14 bodies, 10 joints, 8 actuators)
[PASS] Ball positioning (z_gap=0.0250)
[PASS] Streamer lifecycle (start/stop)
[PASS] mediapy import
[PASS] Correct PID survival (10.0s)
[PASS] Wrong PID fails (0.8s < 2.0s)
[PASS] Joint authority (j5=0.0018, j6=0.0018, j7=0.0001)
[PASS] EGL rendering (480x640x3)
ALL CHECKS PASSED
```

Confirmed: CLAUDE.md does NOT reveal correct joint pairing or sign (physics-only hints).

---

## Verification

1. ~~`scripts/preflight.py`~~ ALL 9 CHECKS PASSED
2. Manual: start `01_validate_assembly.py` in stream mode, verify browser shows live video via VS Code port forwarding
3. Manual: fresh Claude Code session with updated CLAUDE.md converges on correct joints in ≤5 iterations

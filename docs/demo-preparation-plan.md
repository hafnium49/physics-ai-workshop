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

## Step 2: Create the MJPEG live streamer helper

**File:** `mujoco_streamer.py`

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

## Step 3: Update assembly script with live stream (Sprint 1)

**File:** `scripts/01_validate_assembly.py` (update existing)

- Stream live via `LiveStreamer` (with .mp4 fallback)
- `while True` loop with Ctrl+C to stop
- Print diagnostics every second
- **New: joint exploration mode** — accept keyboard/terminal commands to nudge individual joints, so participants build intuition for which joints control what

---

## Step 4: Update baseline PID script (Sprint 2)

**File:** `scripts/02_pid_baseline.py` (update existing)

- Add `LiveStreamer` integration
- `while True` loop with auto-reset on ball fall (reposition ball, restart timer)
- Keep `Survival Time: X.X seconds` terminal output (Claude reads this)
- **Add per-joint authority diagnostics** on each reset: "joint7 correction = 0.05 rad → plate tilt change = 0.001 rad" — gives Claude data to reason from, prevents gain-tuning dead-end spiral
- Deliberate baseline: wrong joints (j6+j7) + wrong sign → 0.8s survival

---

## Step 5: Create disturbance challenge script (Sprint 3 content)

**File:** `scripts/04_challenge.py` (NEW)

Progressive difficulty that restores the tuning arc:

1. **Level 1 — Static hold** (trivial with correct joints): ball stays for 10s. Confirms working PID.
2. **Level 2 — Periodic impulses**: random force pushes on ball every 2s. Gains matter now. Target: survive 10s.
3. **Level 3 — Moving target**: plate tilts in slow sinusoidal circle while keeping ball centered. Target: survive at increasing speeds.
4. **Level 4 — Speed record**: fastest circle speed where ball survives 10s. Competitive target for leaderboard.

Each level visually distinct on the live stream. Terminal prints current level, survival time, max perturbation survived.

---

## Step 6: Run optimization validation

**File:** `scripts/03_optimize_pid.py` (update existing)

Quick validation — already confirmed empirically but needs formal run:
- Test joint pairings × signs → confirm j5+j6 sign=+1 is unique solution
- Test disturbance survival across gain ranges → identify which gains matter for Level 2+
- Print summary table

---

## Step 7: Update workshop materials — HIGHEST PRIORITY after streamer

**`CLAUDE.md`** is dangerously stale and will mislead Claude. Must fix:
- Remove ".mp4 only" constraint → document `mujoco_streamer.py` usage
- Remove disproven 45-degree rotation reference
- Add `panda_ball_balance.xml` documentation
- **Physics-only hint (not full answer):** "The plate is rigidly attached to the end-effector. To tilt the plate, identify which joints produce wrist rotation. Not all joints contribute equally to plate orientation."
- Add Survival Time metric documentation
- Document `MUJOCO_GL=egl` requirement

**`docs/participant-guide.md`** — revised sprint structure:
- Sprint 1 (12 min): Assembly + joint exploration (build intuition)
- Sprint 2 (18 min): PID discovery — human-AI collaboration to find correct joints/sign
- Sprint 3 (25 min): Progressive challenges (impulses → trajectories → speed record)
- Buffer (5 min): wrap-up, questions

**`docs/host-preparation-runbook.md`**:
- No opencv-python needed (using Pillow via mediapy)
- Copy `mujoco_streamer.py` to each `~/physics_sim/`
- Assign ports 8081-8085 per user
- Add smoke test: run streamer, verify VS Code port forwarding

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

## Execution priority

1. **Create `mujoco_streamer.py`** — all script updates depend on it
2. **Update `CLAUDE.md`** — stale file will actively mislead Claude
3. **Update scripts 01, 02 with streaming + fallback**
4. **Create `scripts/04_challenge.py`** — Sprint 3 content
5. **Run `scripts/03_optimize_pid.py`** — formal validation
6. **Update participant-guide.md and host-runbook.md**
7. **End-to-end smoke test** as mock participant

---

## Verification

1. `mujoco_streamer.py` — stream starts, browser shows frames, Ctrl+C stops cleanly
2. `scripts/01_validate_assembly.py` → live stream, ball on plate, joint exploration works
3. `scripts/02_pid_baseline.py` → ball falls off at 0.8s, authority diagnostics print, auto-resets
4. `scripts/04_challenge.py` → Level 1 trivial, Level 2 requires tuning, Level 3+ is hard
5. `--no-stream` fallback produces .mp4 on all scripts
6. Port forwarding works from VS Code Remote SSH client
7. Fresh Claude conversation with updated CLAUDE.md converges on correct joints in ≤5 iterations

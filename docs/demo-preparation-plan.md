# Workshop Demo Preparation Plan

## Context

Build and validate the full demo pipeline for the Physics-AI workshop. Material engineers (non-programmers) use Claude Code to control a Franka Panda arm balancing a ball on a plate via PID control.

**Key pivot:** Real-time MJPEG web streaming replaces .mp4 output. VS Code Remote SSH auto-forwards the port — engineers see live simulation in browser. Iteration time drops from ~90s to ~15s.

**Key physics finding:** With position actuators, PID gain tuning is trivially easy once the correct joints (j6+j7) and sign (+) are found. Static balancing alone won't fill 30 minutes. **Solution: progressive disturbance challenges** restore the difficulty curve.

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
- Interactive camera: left-drag orbit, scroll zoom, right-drag pan, R key reset — via POST `/camera` endpoint + `mjv_moveCamera()` on sim thread
- Scripts use `streamer.make_free_camera(model)` + `streamer.drain_camera_commands()` for interactive camera
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

**File:** `scripts/02_pid_baseline.py` (updated, verified: ~0.3s survival with wrong baseline)

- Add `LiveStreamer` integration
- `while True` loop with auto-reset on ball fall (reposition ball, restart timer)
- Keep `Survival Time: X.X seconds` terminal output (Claude reads this)
- **Add per-joint authority diagnostics** on each reset: "joint7 correction = 0.05 rad → plate tilt change = 0.001 rad" — gives Claude data to reason from, prevents gain-tuning dead-end spiral
- Deliberate baseline: correct joints (j6+j7) but wrong sign → ~0.3s survival
- Ball fall animation: after ball leaves plate, simulation continues until ball hits floor (~0.4s fall + 0.5s settle) before auto-reset

---

## Step 5: Create disturbance challenge script (Sprint 3 content) — DONE

**File:** `scripts/05_challenge.py` (rewritten as controller exploration playground)

Now serves as the participant's editable controller file for Sprint 4. Exports `make_controller(model, dt, home)` so `04_survival_map.py --controller scripts/05_challenge.py` can evaluate it. Has guardrail comments (DO NOT EDIT / EDIT HERE zones) and leveled exploration hints in Japanese.

**File:** `scripts/04_survival_map.py` (standalone survival map metric)

Evaluates controllers via a 20x20 headless grid sweep. Prints Controller Score (mean survival time). Supports `--controller` flag for pluggable controllers. Baseline PID scores ~3.3 sec.

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
| Sprint 3 content | Progressive disturbances + survival map contour | Restores difficulty curve; gains matter under perturbation |
| CLAUDE.md hints | Physics-only (not full answer) | Claude converges in 3-5 iterations |
| Fallback | All scripts: --no-stream → .mp4 via mediapy | Insurance if streaming breaks |

---

## Execution progress

1. ~~Create `mujoco_streamer.py`~~ DONE
2. ~~Update `CLAUDE.md`~~ DONE
3. ~~Update scripts 01, 02 with streaming + fallback~~ DONE (+ `send_frame` → `update` API fix)
4. ~~Create `scripts/05_challenge.py`~~ DONE
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

## Step 11: Workshop restructure — DONE

Restructured from 3 sprints (build from scratch) to 5 sprints (observe → diagnose → understand → experiment → autonomous R&D):
- Sprint 1 (10 min): Explore — run 01_validate_assembly.py, understand the world
- Sprint 2 (10 min): Baseline Diagnosis — run 02_pid_baseline.py (wrong PID), observe and diagnose failure
- Sprint 3 (10 min): Working Controller — run 03_optimize_pid.py, understand why it works vs baseline
- Sprint 4 (15 min): Digital Twin Experiment — measure with 04_survival_map.py, make one change in 05_challenge.py, compare
- Sprint 5 (15 min): Autonomous R&D Loop — Claude self-drives hypothesis/edit/test/compare on 05_challenge.py

Key changes:
- Split 05_challenge.py: Level 4 (survival map) extracted to standalone 04_survival_map.py
- Scripts pre-copied to participant workspaces (spoiler comments kept intentionally)
- Controller Score metric (mean survival time) added to 04_survival_map.py terminal output and contour plot
- Participant guide rewritten with plain-English prompts for Sprint 4
- Host runbook updated to copy scripts directory
- 05_challenge.py rewritten as import-safe controller playground with guardrail comments
- 04_survival_map.py exception handling fixed (BaseException for broken controller resilience)

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
- Copy reference scripts to participant workspaces (spoiler comments kept intentionally for the restructured workshop)

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

## Step 12: Security + operational fixes — DONE

Two reviews (Security Engineer, DevOps Automator) found critical issues.

### Security fixes (physics-ai-workshop, public repo):
- Removed hardcoded `PhysicsAI2026!` from setup script and runbook (replaced with `<WORKSHOP_PASSWORD>` placeholder)
- Added `.claude/settings.local.json` to `.gitignore` (prevents leaking auto-approved commands)
- Bound `mujoco_streamer.py` to `127.0.0.1` instead of `0.0.0.0` (localhost only, VS Code port forwarding still works)

### Operational fixes (physics-ai-workshop):
- Fixed Python venv idempotency: always run pip (was skipping entirely on re-run if venv dir existed)
- Parallelized user provisioning: 5 users concurrently (~5 min vs ~25 min sequential)
- Explicit "password NOT changed" message when skipping existing users

### dgx-spark-playbooks updates (private repo):
- `bootstrap-workshop.sh`: removed global Node.js/Claude Code install, keeps `at` + SSH/security only
- `setup-workshop.sh`: removed user creation (users created by physics-ai-workshop), keeps SSH config/tunnels/dead-man's switch
- `Host_Preparation_Runbook.md`: rewritten Phase 1+2 for per-user provisioning via physics-ai-workshop
- `workshop-ssh-softening-plan.md`: updated scripts table and implementation sequence

### Architecture boundary:
- **dgx-spark-playbooks** (private): SSH tunnels, security hardening, dead-man's switch
- **physics-ai-workshop** (public): user creation, per-user nvm/Node.js/Claude Code, Python venv, workspace files

### Remaining:
- `smoke-test-workshop.sh` still checks for global Claude Code — needs updating to check per-user install

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

---

## Workshop Outcome — 2026-04-01

**ワークショップは成功しました。** 5名の素材エンジニア全員がPIDコントローラの改善に成功。

### 実行結果
- 5スプリント全て予定通り実施（探索→診断→理解→実験→自走型R&D）
- 5名が5通りの異なる制御戦略を開発
- 3名が独立して `data.qvel` による速度直接利用を発見（PID微分項の限界を超える洞察）
- 川原井様は7本の探索スクリプトを自動生成し、完全維持率を73%→91%に向上

### 準備中に発見・修正したバグ
- **Kd=2 問題:** `03_optimize_pid.py` の `make_controller` が Kd=2 を使用 → dt=0.005で400倍増幅 → 0.3秒で落下。Kd=0に修正し、スコアが0.3→3.3に回復
- **AllowTcpForwarding:** SSH Match blockが `no` → VS Code接続不可。`local` に変更
- **動的ポート転送:** VS Code Remote SSHがトンネルチェーンで失敗。`enableDynamicForwarding: false` で解決
- **SSHホストエイリアス衝突:** 全参加者が同じ `workshop-gx10` → per-participant `workshop-gx10-N` に変更

### 参加者の成果詳細
`dgx-spark-playbooks/docs/workshop-results.md` に日本語チューターレビューとして記録済み。

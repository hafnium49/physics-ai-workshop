# Proposal: Physics-AI Workshop as an AI Platform Custom Tool

## 1. Executive Summary

### What this is

A 1-hour hands-on workshop where non-programmers use an AI agent to build and optimize PID controllers for a robotic arm simulation (MuJoCo digital twin). Packaged as a custom tool for a chat-based AI assistant platform using the OpenClaw skill standard.

### Validated results

| Metric | Result |
|--------|--------|
| Participants | 5 material engineers (zero programming experience) |
| Completion rate | 100% — all 5 improved their controllers |
| Discovery rate | 3 of 5 independently discovered an advanced physics insight (qvel state feedback) |
| Time to first improvement | ~35 minutes (Sprint 4) |
| Unique approaches generated | 5 distinct control strategies |

### Why this belongs on the platform

- Demonstrates AI-guided technical education for domain experts
- Reusable pedagogical pattern (5-sprint: observe → diagnose → understand → experiment → autonomous R&D)
- Real physics simulation with measurable outcomes (Controller Score)
- Fully Japanese-language tutor persona for the target audience

---

## 2. Architecture Overview

Two artifacts, cleanly separated:

```
┌─────────────────────────┐     ┌──────────────────────────────┐
│  OpenClaw Skill         │     │  Simulation Service          │
│  (AI instructions)      │     │  (FastAPI + MuJoCo backend)  │
│                         │     │                              │
│  • Tutor persona (JP)   │     │  • MuJoCo physics engine     │
│  • 5-sprint flow        │────▶│  • Script execution          │
│  • Controller templates │     │  • MJPEG live streaming      │
│  • Score analysis       │     │  • Survival map evaluation   │
│  • Sprint gating rules  │     │  • Per-session isolation     │
└─────────────────────────┘     └──────────────────────────────┘
         AI layer                      Execution layer
```

The **skill** tells the AI what to say and which tools to call. The **service** executes the physics and returns results.

---

## 3. Skill Architecture

### Directory structure

```
physics-workshop/
├── SKILL.md                    # Sprint flow, tutor persona, tool call instructions
├── reference.md                # MuJoCo API, joint names, model architecture, physics notes
├── templates/
│   └── controller_template.py  # Starter make_controller() for Sprint 4-5
└── examples/
    ├── pid_baseline.py         # Deliberately broken PID (Sprint 2 discussion)
    └── pid_working.py          # Working PID (Sprint 3 discussion)
```

### SKILL.md frontmatter

```yaml
---
name: physics-workshop
description: >
  MuJoCo物理シミュレーションワークショップ — Franka Pandaロボットアームの
  ボールオンプレートPID制御を構築・改善する。素材エンジニア向け日本語チューター。
  ユーザーがシミュレーション構築、PIDコントローラ作成、関節探索、ライブ配信、
  ボールバランスタスクに取り組む際にアクティブ化。
allowed-tools: >
  sim.create_session, sim.run_script, sim.get_run_status, sim.stop_run,
  sim.upload_controller, sim.get_stream_url, sim.send_camera_command,
  Read(*), Edit(templates/*)
---
```

### Sprint-to-tool mapping

| Sprint | AI reads | AI calls | AI explains |
|--------|----------|----------|-------------|
| 1. 探索 | reference.md (joint names) | `sim.run_script("01_validate_assembly")` → `sim.get_stream_url()` | 世界の構成、関節の役割 |
| 2. 診断 | examples/pid_baseline.py | `sim.run_script("02_pid_baseline")` → `sim.get_run_status()` (survival time) | なぜ失敗するか |
| 3. 理解 | examples/pid_working.py | `sim.run_script("03_optimize_pid")` → `sim.get_stream_url()` | ベースラインとの違い |
| 4. 実験 | templates/controller_template.py | `Edit(templates/...)` → `sim.upload_controller()` → `sim.run_script("04_survival_map")` | スコアの変化と改善理由 |
| 5. 自走R&D | Score history from previous calls | Autonomous loop: edit → upload → evaluate → analyze → repeat | 仮説と実験結果の要約 |

### Sprint gating rules (in SKILL.md body)

```markdown
## 行動指針

- Sprint 1〜3 ではコントローラを編集しない。観察と理解に集中。
- Sprint 4 から `templates/controller_template.py` の編集を開始。
- `sim.upload_controller` は Sprint 4 以降のみ呼び出す。
- 常に `sim.get_stream_url()` の結果をユーザーに提示する（数字だけ出力しない）。
- スコアは「維持マップ」の平均維持時間（秒）。ベースライン ~3.3秒。
```

---

## 4. Tool API Design (Simulation Service)

### Endpoints

#### Session management

```
POST   /api/sessions
  → 201 { session_id, stream_port, workspace_path }

DELETE /api/sessions/{session_id}
  → 204 (kills process, frees port, cleans workspace)
```

#### Script execution

```
POST   /api/sessions/{session_id}/run
  Body: { script: "01_validate_assembly" | "02_pid_baseline" | "03_optimize_pid" | "04_survival_map" | "05_challenge", args?: {} }
  → 202 { run_id, status: "running" }

GET    /api/sessions/{session_id}/runs/{run_id}
  → 200 { status: "running" | "completed" | "failed", stdout_tail: [...], survival_time?: float, controller_score?: float }

POST   /api/sessions/{session_id}/runs/{run_id}/stop
  → 200 { status: "stopped" }
```

#### Live streaming

```
GET    /api/sessions/{session_id}/stream
  → 200 (multipart/x-mixed-replace MJPEG stream — proxied from mujoco_streamer)

POST   /api/sessions/{session_id}/camera
  Body: { action: "rotate" | "pan" | "zoom" | "reset", dx?: float, dy?: float }
  → 200 { ok: true }
```

#### Controller upload

```
POST   /api/sessions/{session_id}/controller
  Body: { source_code: "def make_controller(model, dt, home): ..." }
  → 200 { valid: true, warnings?: [...] }
  → 422 { valid: false, error: "SyntaxError at line 5" }
```

### Response: Controller Score

When `04_survival_map` completes, `get_run_status` returns:

```json
{
  "status": "completed",
  "controller_score": 3.3,
  "perfect_count": 38,
  "total_trials": 400,
  "max_survival": { "time": 10.0, "offset_mm": [69, -69] },
  "stdout_tail": ["スコア: 3.3 秒", "完全維持: 38/400 (9.5%)"]
}
```

---

## 5. UI Integration

### React components

```
┌──────────────────────────────────────────────────────────┐
│  Chat Area                          │  Simulation Panel  │
│                                     │                    │
│  🤖 ワークショップを始めましょう。    │  ┌──────────────┐  │
│  Sprint 1では、ロボットの世界を      │  │              │  │
│  見てみましょう。                    │  │  MJPEG Stream │  │
│                                     │  │  (embedded)   │  │
│  [01_validate_assembly.py を        │  │              │  │
│   実行中...]                        │  └──────────────┘  │
│                                     │                    │
│  ユーザー: 関節6を動かして          │  Sprint: 1/5       │
│                                     │  Score: --         │
│  🤖 関節6はプレートのX方向の傾きを  │                    │
│  制御します。見てください...         │  ┌──────────────┐  │
│                                     │  │ Leaderboard  │  │
│                                     │  │ 川原井: 8.1秒│  │
│                                     │  │ 土田:   7.5秒│  │
│                                     │  │ 石村:   6.8秒│  │
│                                     │  └──────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### Key UI elements

| Component | Data source | Updates |
|-----------|------------|---------|
| MJPEG viewer | `GET /api/sessions/{id}/stream` via `<img>` tag | Continuous |
| Sprint indicator | AI declares sprint transitions in chat | Per sprint |
| Controller Score | `controller_score` from `get_run_status` | After each evaluation |
| Leaderboard | Aggregated scores across active sessions | After each evaluation |
| Camera controls | Mouse events → `POST /api/sessions/{id}/camera` | On user interaction |

### Camera interaction (extracted from mujoco_streamer.py)

```javascript
// Orbit: left-drag
viewer.addEventListener('mousemove', (e) => {
  if (e.buttons === 1) {
    fetch(`/api/sessions/${sessionId}/camera`, {
      method: 'POST',
      body: JSON.stringify({ action: 'rotate', dx: e.movementX, dy: e.movementY })
    });
  }
});

// Zoom: scroll
viewer.addEventListener('wheel', (e) => {
  fetch(`/api/sessions/${sessionId}/camera`, {
    method: 'POST',
    body: JSON.stringify({ action: 'zoom', dy: e.deltaY })
  });
});
```

---

## 6. Environment Requirements

| Requirement | Specification |
|-------------|--------------|
| Python | ≥ 3.10 |
| MuJoCo | ≥ 3.6.0 |
| GPU | Optional (CPU sufficient for single-arm sim; GPU enables MJX acceleration) |
| `MUJOCO_GL` | `egl` (headless rendering via EGL) |
| Packages | `mujoco`, `numpy`, `matplotlib`, `Pillow` |
| Session isolation | Each user gets their own MuJoCo process + working directory |
| Port pool | One port per active session (e.g., 18080-18180) |
| Session timeout | 60 minutes (matching workshop duration) |
| Trial timeout | 15 seconds per simulation trial (prevents infinite loops) |

### Docker image

```dockerfile
FROM python:3.13-slim
RUN apt-get update && apt-get install -y libgl1-mesa-dev libegl1-mesa-dev
RUN pip install mujoco numpy matplotlib Pillow fastapi uvicorn httpx
ENV MUJOCO_GL=egl
COPY content/ /app/content/
COPY scripts/ /app/scripts/
COPY mujoco_streamer.py /app/
```

---

## 7. Migration Path

### Phase 1: Skill only (1 week)

Package SKILL.md + reference.md. AI provides guidance in chat but users execute scripts via terminal (hybrid mode). Validates the tutor persona works on the platform.

**Reuse from repo:**
- `.claude/skills/workshop-sim/SKILL.md` → adapt tool calls
- `CLAUDE.md` → `reference.md`

### Phase 2: Simulation service MVP (2 weeks)

Session lifecycle + script execution + stdout parsing. No streaming — users iterate on Controller Score via text output in chat. Sufficient for Sprints 2-5.

**Reuse from repo:**
- 5 Python scripts → script runner
- `mujoco_streamer.py` → deferred to Phase 3

### Phase 3: Live streaming (2 weeks)

MJPEG proxy + camera commands. Embedded viewer in chat UI. Sprint 1 becomes visually impactful.

**Reuse from repo:**
- `mujoco_streamer.py` → streamer adapter (127.0.0.1 binding, camera command API)

### Phase 4: Multi-user scaling (2-4 weeks)

Kubernetes deployment, GPU scheduling, 20+ concurrent participants, persistent leaderboard.

### Total estimated effort: 7-9 weeks

---

## Appendix A: Controller Interface Contract

Controllers must define `make_controller(model, dt, home)` returning a callable:

```python
def make_controller(model, dt, home):
    """Called once per trial by the evaluator.
    
    Args:
        model: MjModel instance
        dt:    Timestep (0.005s)
        home:  List of 7 home joint angles
    
    Returns:
        controller(data, plate_id, ball_id, step, t) callable
    """
    def controller(data, plate_id, ball_id, step, t):
        # Compute and apply corrections to data.ctrl[5] and data.ctrl[6]
        pass
    return controller
```

Constraints:
- Only `numpy` allowed (no new dependencies)
- Must set `data.ctrl[5]` (joint6, X-axis) and `data.ctrl[6]` (joint7, Y-axis)
- Other joints (0-4, 7) are held at home by the evaluator
- Must complete within 15 seconds per 10-second trial

## Appendix B: Controller Score Methodology

**Controller Score** = mean survival time (seconds) across a 20×20 grid of initial ball positions.

- Grid range: ±120mm from plate center in X and Y
- Per-position trial: 10 seconds maximum
- Ball-off detection: `|error_x| > 140mm` or `|error_y| > 140mm` or ball Z drops below plate
- Baseline score: ~3.3 seconds (P-only controller, Kp=2.0)

Score interpretation:

| Score | Level | Typical approach |
|-------|-------|-----------------|
| < 3.3s | Below baseline | Wrong joints, wrong sign, or excessive gains |
| 3.3-4.0s | Marginal | Basic gain tuning |
| 4.0-5.0s | Good | Derivative term or velocity feedback |
| 5.0-8.0s | Excellent | State feedback (qvel), output limiting, or gain scheduling |
| > 8.0s | Outstanding | Composite control strategies |

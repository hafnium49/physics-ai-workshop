# Workshop Demo Preparation Plan

## Context

Build and validate the full demo pipeline for the Physics-AI workshop. Material engineers (non-programmers) use Claude Code to control a Franka Panda arm balancing a ball on a plate via PID control. The "Autonomous Scientist" concept: Claude writes PID code, runs the sim, reads terminal metrics (Survival Time), and iterates autonomously.

Three independent reviews confirmed: the physics is feasible, but the primary risk is model assembly (collision flags, attachment method, joint ranges), not PID tuning.

---

## Step 1: Create the combined world XML

**File:** `content/panda_ball_balance.xml`

**Primary approach: Rigid kinematic attachment** (NOT weld constraint). Copy `panda.xml` content into a new combined file and nest the plate body directly inside the `<body name="hand">` element. This avoids soft-constraint wobble that would make PID tuning unreliable.

Key details:
- Copy scene setup from `scene.xml` (ground, lighting, skybox, `timestep=0.005`, `implicitfast`)
- Plate body as child of `hand`, at `pos="0.005 0.003 0.1"` (small XY offset = natural tilt)
- Ball as child of plate with free joint, at `pos="0 0 0.025"` (resting on surface)
- Explicit friction on plate and ball geoms: `friction="1 0.005 0.0001"`
- **Critical: plate/ball geoms must have `contype="1" conaffinity="1"`** (Panda collision meshes default to `contype="0"`)
- Add a fixed camera: `<camera name="side" pos="1.0 -0.5 1.0" lookat="0.3 0 0.6"/>` for consistent video framing

**Joint6 range issue:** Range is `[-0.0175, 3.7525]` — nearly zero room in the negative direction. Set home position for joint6 to ~1.8 rad (mid-range) instead of π/2 to ensure symmetric control authority in both tilt directions.

**Tradeoff:** Copying panda.xml means we don't auto-inherit upstream changes, but for a workshop this is fine. Modularity matters less than physics stability.

---

## Step 2: Validate assembly (Sprint 1 demo)

**File:** `scripts/01_validate_assembly.py`

- Load `content/panda_ball_balance.xml`
- Set initial arm home pose: `j1=0, j2=-0.785, j3=0, j4=-2.356, j5=0, j6=1.8, j7=0.785`
  - Note: j6=1.8 (mid-range), not π/2, to address asymmetric joint range
- Hold arm via position actuators (`data.ctrl[0:7]` = target joint angles)
- Run 3 seconds, render at 30fps using the fixed `side` camera → `assembly_test.mp4`
- Print diagnostics: plate/ball positions, whether ball is on plate, check for NaN
- **Validation criteria:**
  - No plate jitter or vibration (rigid attachment should eliminate this)
  - Ball visible on plate at start
  - Ball stays on plate at least briefly (confirms collision is working — ball doesn't fall through)
  - No NaN in body positions

---

## Step 3: Baseline PID controller (Sprint 2 demo)

**File:** `scripts/02_pid_baseline.py`

PID architecture:
- **Sense:** Ball XY position relative to plate center (`data.xpos[ball_id] - data.xpos[plate_id]`)
- **Transform:** Rotate error vector by ~45 degrees to account for the hand's `quat="0.9238795 0 0 -0.3826834"` rotation. Without this, X/Y errors map to diagonal joint corrections.
- **Control:** Rotated X-error → joint6 correction, rotated Y-error → joint7 correction
- **Actuate:** `data.ctrl[5] = nominal_j6 + pid_x.compute(rotated_error_x)`, same for ctrl[6]/joint7
- All other joints held at home position

Failure detection:
```
ball off plate when: |ball_rel_x| > 0.14 or |ball_rel_y| > 0.14 or ball_rel_z < -0.02
```

Print `Survival Time: X.X seconds` to terminal — this is the metric Claude reads.

Start with deliberately mediocre gains: `Kp=50, Kd=10, Ki=0` → expect 1-3 second survival.

Also print per-step diagnostics: `Ball X error = +0.05, joint6 correction = +0.02` (for debugging axis mapping).

Add NaN guard and max-time cap (10 seconds).

Save video as `attempt_1.mp4`.

---

## Step 4: Validate optimization (Sprint 3 proof)

**File:** `scripts/03_optimize_pid.py`

Purpose: Prove that a 10-second solution EXISTS in the physics, via grid search. (During the actual workshop, Claude iterates interactively, not via grid search.)

- `run_trial(kp, kd, ki=0) → survival_time` function (no video rendering in search loop)
- Grid search: Kp in [20, 50, 100, 200, 500], Kd in [2, 5, 10, 20, 50]
- Print each result: `Kp=100, Kd=20 -> Survival Time: 7.3s`
- NaN guard per trial (skip divergent trials, don't crash the search)
- Render final video only for best params → `best_balance.mp4`
- If no combo reaches 10s, also try adding Ki=[0, 1, 5] for steady-state offset correction

---

## Step 5: Update workshop materials

- **`CLAUDE.md`**: Add `panda_ball_balance.xml` documentation, Survival Time metric, axis rotation note, joint6 mid-range home position
- **`docs/participant-guide.md`**: Revise Sprint 3 structure:
  - Phase A (10 min): "Autonomous Scientist" demo — Claude runs 3-4 iterations autonomously
  - Phase B (15 min): Human-AI collaboration — engineer directs Claude based on physical intuition ("the ball rolls left, increase stiffness")
  - Phase C (5 min): Victory lap — before/after comparison video
- Add prompt templates for each sprint phase
- Add "break glass" fallback note: if Sprint 2 isn't working by 25-min mark, provide pre-built baseline script

---

## Key technical decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Plate attachment | Rigid kinematic (child of `hand`) | Eliminates weld wobble; PID can trust plate angle |
| PID joints | joint6 + joint7 | Wrist pitch/roll, closest to end-effector |
| Joint6 home | 1.8 rad (mid-range) | Asymmetric range [-0.0175, 3.7525] needs centered nominal |
| Error rotation | ~45 deg rotation of XY error | Hand has 45-deg Z rotation; naive mapping gives diagonal control |
| Ball initial state | Resting on plate | No drop impulse; isolates rolling dynamics |
| Collision flags | contype=1, conaffinity=1 on plate/ball | Panda defaults to contype=0; ball would fall through |

---

## Verification

1. `python scripts/01_validate_assembly.py` → `assembly_test.mp4`: plate attached, ball visible, no jitter, no NaN
2. `python scripts/02_pid_baseline.py` → `attempt_1.mp4` + prints `Survival Time: ~1-3s`
3. `python scripts/03_optimize_pid.py` → finds gains achieving 10s, renders `best_balance.mp4`
4. All scripts headless, output .mp4 via mediapy only
5. Empirically verify axis mapping: positive X error → correct joint correction direction

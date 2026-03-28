# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

This is a 1-hour hands-on workshop where material engineers (not software developers) use Claude Code to build MuJoCo digital twins on an NVIDIA DGX Spark with Blackwell GPU. No programming experience is assumed.

## Environment

- Python virtual environment `workshop_env` is pre-activated (if not: `source ~/workshop_env/bin/activate`)
- Available packages: `mujoco`, `mediapy`, `numpy`
- Participant working directory: `~/physics_sim/` (content files are copied there)
- GPU: NVIDIA Blackwell (available for MuJoCo MJX GPU-accelerated simulation)
- `MUJOCO_GL=egl` environment variable is required for headless rendering
- `mujoco_streamer.py` helper is available in the workspace for live visualization

## Available Models

- `content/franka_panda/panda.xml` — 7-DOF arm from MuJoCo Menagerie
- `content/franka_panda/scene.xml` — Panda with default scene (ground plane, lighting)
- `content/ball_and_plate.xml` — Flat plate + ball with free joint (standalone fragment)
- `content/panda_ball_balance.xml` — Pre-assembled Panda arm gripping a plate (with ball as free body). Ready for PID balancing task.

## Constraints

- **No GUI** — users connect via VS Code Remote SSH. Never attempt to open viewer windows.
- **Primary output is live streaming** via `mujoco_streamer.py`. Fall back to `.mp4` via `mediapy.write_video()` only if streaming is unavailable.
- **Keep explanations simple** — avoid programming jargon. Explain physics/control concepts when introducing them.
- **Always include visual output** in simulation scripts so the user can see results.
- Use MuJoCo's Python bindings (`import mujoco`).

## Workshop Goal

Build a Franka Panda arm holding a plate with a ball, then optimize PID control to keep the ball balanced for 10 seconds.

### Sprint Structure

1. **Explore** (15 min) — Load the pre-assembled model, start live stream, move joints to build intuition
2. **PID Discovery** (15 min) — Write a PID controller, discover which joints actually tilt the plate
3. **Progressive Challenges** (30 min) — Add disturbances, tune gains under perturbation, push difficulty

## Live Visualization

Use the `mujoco_streamer.py` helper for real-time browser-based viewing:

```python
from mujoco_streamer import LiveStreamer

streamer = LiveStreamer()
streamer.start()

# In simulation loop:
renderer.update_scene(data, camera="side")
streamer.update(renderer.render())
```

Each participant has a unique streaming port assigned via the `STREAM_PORT` environment variable. `LiveStreamer()` reads this automatically — do not hardcode a port number.

VS Code automatically detects the port and offers "Open in Browser".
Fall back to `mediapy.write_video()` only if streaming is unavailable.

## Ball-on-Plate Balancing Task

The pre-assembled model `panda_ball_balance.xml` has:
- Plate rotated 90° and positioned so the gripper fingers clamp its edge (`ctrl[7]=0.008`). The plate extends horizontally outward from the grip point.
- Ball as a top-level free body — must be repositioned onto the plate in scripts

**Important physics notes:**
- The plate is rigidly attached to the end-effector and extends outward from the grip. To tilt the plate, identify which wrist joints produce rotation in the plate plane. Not all joints contribute equally to plate orientation.
- The ball has a free joint and can roll/fall off the plate.
- Use `data.xpos[ball_id] - data.xpos[plate_id]` to track ball position relative to plate.
- Print `Survival Time: X.X seconds` to terminal for optimization tracking.

**Ball repositioning (required in every script):**
After setting arm joint positions and calling `mj_forward()`, place the ball on the plate:

```python
ball_qpos_addr = model.jnt_qposadr[ball_joint_id]
data.qpos[ball_qpos_addr:ball_qpos_addr+3] = data.xpos[plate_id] + [0, 0, 0.025]
data.qpos[ball_qpos_addr+3:ball_qpos_addr+7] = [1, 0, 0, 0]  # identity quaternion
```

**Home pose:** `j1=0, j2=-0.785, j3=0, j4=-2.356, j5=1.184, j6=3.184, j7=1.158`
Note: j5, j6, j7 are set so the plate is horizontal with the edge gripped by the fingers.

When writing a PID controller for the first time, do not run a systematic joint authority analysis upfront. Let the initial attempt use a reasonable guess for which joints to control, and diagnose from the results.

## Model Architecture

### Franka Panda (`content/franka_panda/`)
- `panda.xml` — 7-DOF arm from MuJoCo Menagerie. Kinematic chain: `link0` -> `link1` ... -> `link7` -> `hand` -> `left_finger`/`right_finger`
- `scene.xml` — Includes `panda.xml` plus ground plane, lighting, skybox. Uses `timestep="0.005"` with `implicitfast` integrator.
- **End-effector attachment point:** The `hand` body (child of `link7`) has a site named `gripper` at `pos="0 0 0.1"`. In `panda_ball_balance.xml`, the plate is rotated 90° and positioned so the fingers grip its edge.
- **Actuators:** 7 position-controlled joints (`actuator1`-`actuator7`) + 1 gripper actuator (`actuator8`). Joints use built-in PD control with `kp` and `kv` gains already set.
- **Joint names:** `joint1`-`joint7` (arm), `finger_joint1`/`finger_joint2` (gripper, coupled via equality constraint)

### Ball and Plate (`content/ball_and_plate.xml`)
- `plate` body: box geom `0.15 x 0.15 x 0.005`, mass 0.5 kg
- `ball` body: sphere radius 0.02, mass 0.1 kg, has a `free` joint (`ball_free`) — 6-DOF unconstrained motion
- To assemble: nest the plate body inside the Panda's `hand` body with a 90° rotation (`quat="0.7071 0.7071 0 0"`) so the fingers grip the plate edge. ball_and_plate.xml is a standalone fragment — do not use `<include>`.

## MuJoCo Quick Reference

```python
import mujoco
import numpy as np
from mujoco_streamer import LiveStreamer

model = mujoco.MjModel.from_xml_path("path/to/model.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

# Live streaming setup
streamer = LiveStreamer()
streamer.start()

for _ in range(duration_steps):
    mujoco.mj_step(model, data)
    renderer.update_scene(data)
    streamer.update(renderer.render())

# Fallback: save to file if streaming is unavailable
# import mediapy
# frames = []
# for _ in range(duration_steps):
#     mujoco.mj_step(model, data)
#     renderer.update_scene(data)
#     frames.append(renderer.render())
# mediapy.write_video("output.mp4", frames, fps=30)
```

### Useful APIs for this workshop
- `data.qpos` / `data.qvel` — joint positions and velocities
- `data.ctrl` — set actuator control signals
- `mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")` — look up body/joint/site IDs by name
- `data.xpos[body_id]` — Cartesian position of a body after `mj_step`
- `model.opt.timestep` — simulation timestep (0.005s in scene.xml = 200 Hz)

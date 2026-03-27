# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

This is a 1-hour hands-on workshop where material engineers (not software developers) use Claude Code to build MuJoCo digital twins on an NVIDIA DGX Spark with Blackwell GPU. No programming experience is assumed.

## Environment

- Python virtual environment `workshop_env` is pre-activated (if not: `source ~/workshop_env/bin/activate`)
- Available packages: `mujoco`, `mediapy`, `numpy`
- Participant working directory: `~/physics_sim/` (content files are copied there)
- GPU: NVIDIA Blackwell (available for MuJoCo MJX GPU-accelerated simulation)

## Constraints

- **No GUI** — users connect via VS Code Remote SSH. Always render simulations to `.mp4` files using `mediapy.write_video()`. Never attempt to open viewer windows.
- **Keep explanations simple** — avoid programming jargon. Explain physics/control concepts when introducing them.
- **Always include video output** in simulation scripts so the user can see results.
- Use MuJoCo's Python bindings (`import mujoco`).

## Workshop Goal

Build a Franka Panda arm holding a plate with a ball, then optimize PID control to keep the ball balanced for 10 seconds.

### Sprint Structure

1. **Assembly** (15 min) — Combine models, attach plate to end-effector, render to video
2. **Baseline** (15 min) — Drop ball onto plate, apply basic PID controller, record results
3. **Optimization** (30 min) — Systematically tune Kp/Kd until ball stays centered for 10 seconds

## Model Architecture

### Franka Panda (`content/franka_panda/`)
- `panda.xml` — 7-DOF arm from MuJoCo Menagerie. Kinematic chain: `link0` → `link1` ... → `link7` → `hand` → `left_finger`/`right_finger`
- `scene.xml` — Includes `panda.xml` plus ground plane, lighting, skybox. Uses `timestep="0.005"` with `implicitfast` integrator.
- **End-effector attachment point:** The `hand` body (child of `link7`) has a site named `gripper` at `pos="0 0 0.1"` — attach the plate here.
- **Actuators:** 7 position-controlled joints (`actuator1`–`actuator7`) + 1 gripper actuator (`actuator8`). Joints use built-in PD control with `kp` and `kv` gains already set.
- **Joint names:** `joint1`–`joint7` (arm), `finger_joint1`/`finger_joint2` (gripper, coupled via equality constraint)

### Ball and Plate (`content/ball_and_plate.xml`)
- `plate` body: box geom `0.15 x 0.15 x 0.005`, mass 0.5 kg
- `ball` body: sphere radius 0.02, mass 0.1 kg, has a `free` joint (`ball_free`) — 6-DOF unconstrained motion
- To assemble: nest the plate/ball bodies inside the Panda's `hand` body (or weld to `gripper` site) rather than using `<include>`, since ball_and_plate.xml is a standalone fragment

## MuJoCo Quick Reference

```python
import mujoco
import mediapy
import numpy as np

model = mujoco.MjModel.from_xml_path("path/to/model.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

frames = []
for _ in range(duration_steps):
    mujoco.mj_step(model, data)
    renderer.update_scene(data)
    frames.append(renderer.render())

mediapy.write_video("output.mp4", frames, fps=30)
```

### Useful APIs for this workshop
- `data.qpos` / `data.qvel` — joint positions and velocities
- `data.ctrl` — set actuator control signals
- `mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")` — look up body/joint/site IDs by name
- `data.xpos[body_id]` — Cartesian position of a body after `mj_step`
- `model.opt.timestep` — simulation timestep (0.005s in scene.xml = 200 Hz)

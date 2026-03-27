# CLAUDE.md

## Context

This is a hands-on Physics-AI workshop. The user is a material engineer (not a software developer) building MuJoCo digital twins using Claude Code on an NVIDIA DGX Spark with Blackwell GPU.

## Environment

- Python virtual environment is pre-activated (`workshop_env`)
- Available packages: `mujoco`, `mediapy`, `numpy`
- Working directory: `~/physics_sim/`
- Available models:
  - `content/franka_panda/panda.xml` — Franka Emika Panda 7-DOF arm (from MuJoCo Menagerie)
  - `content/franka_panda/scene.xml` — Panda with a default scene (ground plane, lighting)
  - `content/ball_and_plate.xml` — flat plate + ball with free joint
- GPU: NVIDIA Blackwell (available for MuJoCo MJX GPU-accelerated simulation)

## Constraints

- **No GUI** — the user is connected via SSH. Always render simulations to `.mp4` files using `mediapy`.
- **Keep explanations simple** — the user is a material engineer, not a software developer. Avoid jargon; explain concepts when needed.
- When writing simulation scripts, always include video output so the user can see results.
- Use MuJoCo's Python bindings (`import mujoco`).

## Workshop Goal

Build a Franka Panda arm holding a plate with a ball, then optimize PID control to keep the ball balanced for 10 seconds.

### Sprint Structure

1. **Assembly** (15 min) — Combine the Panda arm XML with ball_and_plate.xml. Attach the plate to the robot's end-effector. Run a simulation and render to video.
2. **Baseline** (15 min) — Drop the ball onto the plate. Apply a basic PID controller. Observe how the ball behaves. Save as video.
3. **Optimization** (30 min) — Tune Kp and Kd values until the ball stays centered on the plate for 10 seconds. Iterate systematically. Record the best result.

## MuJoCo Quick Reference

```python
import mujoco
import mediapy
import numpy as np

# Load model
model = mujoco.MjModel.from_xml_path("path/to/model.xml")
data = mujoco.MjData(model)

# Create renderer
renderer = mujoco.Renderer(model, height=480, width=640)

# Simulation loop with video capture
frames = []
for _ in range(duration_steps):
    mujoco.mj_step(model, data)
    renderer.update_scene(data)
    frames.append(renderer.render())

# Save video
mediapy.write_video("output.mp4", frames, fps=30)
```

---
name: workshop-sim
description: MuJoCo physics simulation workshop — procedural guidance for building and tuning a PID controller for ball-on-plate balancing with a Franka Panda robot arm. Activates when users build simulations, write PID controllers, explore joints, stream live video, or troubleshoot ball-balancing tasks.
---

# Workshop Simulation Skill

Procedural guidance for the ball-on-plate balancing workshop. See CLAUDE.md for reference data (home pose, model paths, API).

## Simulation Script Template

Every script follows three phases:

### Phase 1 — Setup
1. Load model with `MjModel.from_xml_path()`
2. Set all 7 arm joints to the home pose (see CLAUDE.md)
3. Set gripper: `data.ctrl[7] = 0.008`
4. Call `mj_forward()` to compute positions
5. Reposition ball onto the plate (see CLAUDE.md ball repositioning snippet)
6. Create renderer and start `LiveStreamer()` (reads `STREAM_PORT` env var automatically)

### Phase 2 — Control Loop
```
for each timestep:
    mj_step()
    hold non-PID joints at home pose
    compute ball error = xpos[ball] - xpos[plate]
    compute PID correction from error (P and optionally D terms)
    apply correction: ctrl[joint_index] = home[joint_index] + correction
    render and push frame to streamer
```

### Phase 3 — Termination and Reset
- Detect ball off plate: `abs(error_x) > 0.14` or `abs(error_y) > 0.14` or ball Z drops below plate
- Print `Survival Time: X.X seconds` to terminal
- Auto-reset: reposition ball, zero velocities, restart the loop

## Sprint 1: Exploration

Write a "just observe" script — load the model, hold the home pose, stream live. No controller. Watch the ball fall off naturally.

**Single-joint isolation procedure:**
1. Pick one wrist joint
2. Offset it from home by a small amount (e.g., 0.1 radians)
3. Hold all other joints at home
4. Call `mj_forward()` and observe how the plate moves
5. Repeat for each wrist joint independently
6. Note which joints move the plate strongly vs. barely at all

## Sprint 2: PID Discovery Cycle

Follow this iterative process. Do NOT run diagnostics as the first step — write a PID attempt first and let it fail naturally.

1. **Pick joints**: Choose wrist joints you think control plate tilt
2. **Write PID**: Apply proportional (and optionally derivative) correction to those joints
3. **Run and observe**: Watch the stream, read the survival time
4. **If survival time is short** (<2 seconds):
   - Run the nudge diagnostic: perturb each candidate joint by a small amount, call `mj_forward()`, measure how much the plate position changes
   - Switch to the joints that produce the largest plate displacement
5. **If behavior is inverted** (ball accelerates off instead of correcting):
   - The correction sign is wrong — flip it
6. **Re-run and compare** survival time to the previous attempt
7. **Repeat** until the ball survives 10 seconds

## Common Failure Patterns

| Symptom | Diagnostic Step |
|---------|----------------|
| Ball flies off immediately (<0.5s) | Check the correction sign — it may be pushing the ball off instead of correcting |
| Ball drifts slowly off one edge | Check if you are controlling the correct axis for that direction |
| Ball oscillates wildly on the plate | Gains may be too high — try reducing Kp significantly |
| NaN errors in simulation | Correction values may be exploding — add output clamping or reduce gains |
| Survival time stuck at ~1s regardless of gains | You may be controlling the wrong joints entirely — run the nudge diagnostic |
| "Port already in use" error | A previous script is still running — press Ctrl+C in that terminal first |

## Sprint 3: Challenge Patterns

### Impulse Disturbances
Apply random force pushes to the ball using `data.xfrc_applied[ball_id, :3]`. Apply the force for a short burst (e.g., 10 timesteps), then zero it out. Repeat at regular intervals.

### Sinusoidal Plate Oscillation
Add a time-varying offset to the PID joint commands:
```
offset = amplitude * sin(2 * pi * frequency * t)
ctrl[joint] = home[joint] + pid_correction + offset
```

### Comparative Experiments
Run multiple parameter values, record survival time for each, report the best. Useful for finding the gain sweet spot or the maximum disturbance force the controller can handle.

## Streaming and Interactive Camera

- Create a free camera with `cam = streamer.make_free_camera(model)` after `streamer.start()`
- In the render loop, call `streamer.drain_camera_commands(model, cam, renderer.scene)` before `renderer.update_scene(data, camera=cam)` — this enables browser-based orbit/zoom/pan
- Use `render_every = int(1.0 / (fps * dt))` to limit rendering to ~30fps
- Always wrap the simulation in `try/finally` with `streamer.stop()` in the finally block
- Do NOT call `renderer.render()` from a thread other than the simulation thread (OpenGL is not thread-safe)
- `LiveStreamer()` with no port argument reads `STREAM_PORT` env var automatically — do not hardcode port numbers

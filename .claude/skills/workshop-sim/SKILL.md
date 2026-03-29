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

## Sprint 3: Challenges (8 min)

Participants run `05_challenge.py` to see disturbances in action. This is a quick demo, not the main activity.

### Impulse Disturbances
Apply random force pushes to the ball using `data.xfrc_applied[ball_id, :3]`. Apply the force for a short burst (e.g., 10 timesteps), then zero it out. Repeat at regular intervals.

### Sinusoidal Plate Oscillation
Add a time-varying offset to the PID joint commands:
```
offset = amplitude * sin(2 * pi * frequency * t)
ctrl[joint] = home[joint] + pid_correction + offset
```

## Sprint 4: Free Exploration (25 min)

Participants edit `scripts/05_challenge.py` (the controller playground) to improve the controller, then evaluate with `scripts/04_survival_map.py`.

### Workflow
1. Participant asks Claude to improve the controller in `05_challenge.py`
2. Claude edits the `make_controller()` function inside `05_challenge.py`
3. Participant runs: `python scripts/04_survival_map.py --controller scripts/05_challenge.py`
4. The survival map shows the Controller Score — compare with baseline (~3.3 sec)
5. Repeat

### Approaches to suggest when asked to improve the controller
- **Velocity feedback**: Enable the derivative term (Kd > 0) to react to ball speed, not just position
- **Gain tuning**: Systematically try different Kp and Kd values, compare survival maps
- **Asymmetric gains**: Use different gains for X vs Y directions
- **Gain scheduling**: Use stronger corrections when the ball is far from center, gentler when close
- **Integral correction**: Add a small integral term (Ki) to fix persistent drift
- **Output clamping**: Limit maximum correction to prevent overreaction and instability

### How to evaluate improvements
The survival map prints a **Controller Score** = mean survival time in seconds across all grid positions. Baseline PID (Kp=2, Kd=0) scores ~3.3 sec. Higher is better. The score also appears on the contour plot.
- Score < 3.3: worse than baseline PID
- Score 3.3-4.0: marginal improvement
- Score 4.0-5.0: meaningful improvement (better gains or derivative term)
- Score > 5.0: significant improvement (likely requires a different control approach)

### Controller file interface
`05_challenge.py` exports `make_controller(model, dt, home)`:
```python
def make_controller(model, dt, home):
    """Called once per trial. Return a controller function."""
    def controller(data, plate_id, ball_id, step, t):
        # Compute and apply corrections to data.ctrl
        pass
    return controller
```
Evaluate with: `python scripts/04_survival_map.py --controller scripts/05_challenge.py`
Always stream the result to the browser — avoid `--no-stream`.

### Important notes
- Edit `05_challenge.py` — do NOT modify `04_survival_map.py`
- Let the participant describe what they want in plain English — do NOT require control theory jargon
- Always run the survival map after making changes so the participant can see the effect
- If an approach fails to implement, fall back to gain tuning (systematically try many Kp/Kd values)
- Controllers must use only numpy — do not suggest importing new dependencies

## Streaming and Interactive Camera

- Create a free camera with `cam = streamer.make_free_camera(model)` after `streamer.start()`
- In the render loop, call `streamer.drain_camera_commands(model, cam, renderer.scene)` before `renderer.update_scene(data, camera=cam)` — this enables browser-based orbit/zoom/pan
- Use `render_every = int(1.0 / (fps * dt))` to limit rendering to ~30fps
- Always wrap the simulation in `try/finally` with `streamer.stop()` in the finally block
- Do NOT call `renderer.render()` from a thread other than the simulation thread (OpenGL is not thread-safe)
- `LiveStreamer()` with no port argument reads `STREAM_PORT` env var automatically — do not hardcode port numbers

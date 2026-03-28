"""Step 3: Baseline PID controller for ball-on-plate balancing.

Applies a PID controller to joint6/joint7 to keep the ball centered on the plate.
Prints Survival Time to terminal. Deliberately uses WRONG SIGN so the ball falls off quickly.

Default: live MJPEG streaming with auto-reset on ball drop.
Fallback: --no-stream saves a single 10s attempt as .mp4.

Run with: python scripts/02_pid_baseline.py [--no-stream] [--port 18080] [--kp 50] [--kd 10]
"""
import os
import sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import mujoco
import numpy as np

try:
    from mujoco_streamer import LiveStreamer
    HAS_STREAMER = True
except ImportError:
    HAS_STREAMER = False


class PIDController:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error):
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


# --- CLI arguments ---
parser = argparse.ArgumentParser(description="PID baseline for ball-on-plate balancing")
parser.add_argument("--no-stream", action="store_true",
                    help="Disable live streaming; save .mp4 instead")
parser.add_argument("--port", type=int, default=18080,
                    help="Streaming port (default: 18080)")
parser.add_argument("--kp", type=float, default=50.0,
                    help="Proportional gain (default: 50.0)")
parser.add_argument("--kd", type=float, default=10.0,
                    help="Derivative gain (default: 10.0)")
args = parser.parse_args()

KP = args.kp
KI = 0.0
KD = args.kd

# --- Setup ---
model = mujoco.MjModel.from_xml_path("content/panda_ball_balance.xml")
data = mujoco.MjData(model)
dt = model.opt.timestep

plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")

# Home pose (j5, j6, j7 adjusted for new plate orientation)
home = [0.0, -0.785, 0.0, -2.356, 1.184, 3.184, 1.158]
joint_names = [f"joint{i}" for i in range(1, 8)]

# Joint IDs and addresses (cached for diagnostics)
joint_ids = {}
for jn in joint_names:
    joint_ids[jn] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)

ball_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
ball_qpos_addr = model.jnt_qposadr[ball_joint_id]
ball_qvel_addr = model.jnt_dofadr[ball_joint_id]

# Renderer
renderer = mujoco.Renderer(model, height=480, width=640)
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side")


def reset_scene(m, d):
    """Reset arm to home pose and place ball on plate."""
    mujoco.mj_resetData(m, d)
    for jn, val in zip(joint_names, home):
        jid = joint_ids[jn]
        d.qpos[m.jnt_qposadr[jid]] = val
    for i, val in enumerate(home):
        d.ctrl[i] = val
    d.ctrl[7] = 0.008  # close gripper
    mujoco.mj_forward(m, d)

    # Place ball on plate
    d.qpos[ball_qpos_addr:ball_qpos_addr + 3] = d.xpos[plate_id] + [0, 0, 0.025]
    d.qpos[ball_qpos_addr + 3:ball_qpos_addr + 7] = [1, 0, 0, 0]
    d.qvel[ball_qvel_addr:ball_qvel_addr + 6] = 0
    mujoco.mj_forward(m, d)


def run_joint_diagnostics(m, d):
    """Per-joint authority diagnostics for wrist joints 5, 6, 7.

    Nudge each wrist joint by +0.01 rad and measure plate position change.
    This helps identify which joints actually tilt the plate.
    Correct answer: joint6 (ctrl[5]) tilts X, joint7 (ctrl[6]) tilts Y.
    """
    # Save current state
    qpos_save = d.qpos.copy()
    qvel_save = d.qvel.copy()
    ctrl_save = d.ctrl.copy()

    # Get baseline plate position
    mujoco.mj_forward(m, d)
    plate_pos_base = d.xpos[plate_id].copy()

    parts = []
    for jnum in [5, 6, 7]:
        jname = f"joint{jnum}"
        jid = joint_ids[jname]
        qadr = m.jnt_qposadr[jid]

        # Nudge
        d.qpos[:] = qpos_save
        d.qvel[:] = qvel_save
        d.ctrl[:] = ctrl_save
        d.qpos[qadr] += 0.01
        mujoco.mj_forward(m, d)

        plate_pos_nudged = d.xpos[plate_id].copy()
        dx = plate_pos_nudged[0] - plate_pos_base[0]
        dy = plate_pos_nudged[1] - plate_pos_base[1]
        dxy = np.sqrt(dx**2 + dy**2)

        parts.append(f"joint{jnum} +0.01 rad → plate dX={dx:+.4f} dY={dy:+.4f} dXY={dxy:.4f}")

    # Restore state
    d.qpos[:] = qpos_save
    d.qvel[:] = qvel_save
    d.ctrl[:] = ctrl_save
    mujoco.mj_forward(m, d)

    print(f"[DIAG] {parts[0]}  |  {parts[1]}  |  {parts[2]}")


def ball_off_plate(d):
    """Check if ball has fallen off the plate."""
    ball_rel_world = d.xpos[ball_id] - d.xpos[plate_id]
    error_x = ball_rel_world[0]
    error_y = ball_rel_world[1]
    return (abs(error_x) > 0.14 or abs(error_y) > 0.14 or ball_rel_world[2] < -0.02)


def run_simulation_step(d, pid_x, pid_y):
    """Run one simulation step with PID control. Returns (error_x, error_y, nan_detected)."""
    mujoco.mj_step(model, d)

    # Hold non-PID joints at home
    for i in [0, 1, 2, 3, 4]:  # joint1-5
        d.ctrl[i] = home[i]
    d.ctrl[7] = 0.008  # gripper

    # Sense: ball position relative to plate in world frame
    ball_rel_world = d.xpos[ball_id] - d.xpos[plate_id]
    error_x = ball_rel_world[0]
    error_y = ball_rel_world[1]

    # NaN guard
    if np.any(np.isnan(d.xpos[ball_id])):
        return error_x, error_y, True

    correction_x = pid_x.compute(error_x)
    correction_y = pid_y.compute(error_y)

    # Apply corrections to joint6 (ctrl[5]) and joint7 (ctrl[6])
    # ===== DELIBERATE BASELINE BUG =====
    # This uses the correct joints BUT the WRONG sign (negative).
    # - The negative sign pushes the plate the wrong way, so the ball
    #   rolls off almost immediately.
    # The workshop task is for Claude to discover:
    #   1. ctrl[5]=joint6 for X, ctrl[6]=joint7 for Y (already correct)
    #   2. The correction sign should be POSITIVE, not negative
    #   3. Kp~2, Kd~0 is sufficient once the sign is fixed
    # With correct sign and moderate gains, the ball survives 10s.
    # ===================================
    d.ctrl[5] = home[5] - correction_x  # joint6 for X (wrong sign!)
    d.ctrl[6] = home[6] - correction_y  # joint7 for Y (wrong sign!)

    return error_x, error_y, False


# ============================================================
# .mp4 mode: single 10s run, save video, exit
# ============================================================
if args.no_stream or not HAS_STREAMER:
    import mediapy

    if not args.no_stream and not HAS_STREAMER:
        print("WARNING: mujoco_streamer not installed, falling back to .mp4 output")

    reset_scene(model, data)
    run_joint_diagnostics(model, data)

    pid_x = PIDController(KP, KI, KD, dt)
    pid_y = PIDController(KP, KI, KD, dt)

    duration = 10.0
    fps = 30
    steps = int(duration / dt)
    render_every = int(1.0 / (fps * dt))
    frames = []
    survival_time = duration

    print(f"PID gains: Kp={KP}, Ki={KI}, Kd={KD}")
    print(f"Simulating {duration}s...")
    print()

    for step in range(steps):
        error_x, error_y, nan_detected = run_simulation_step(data, pid_x, pid_y)

        if nan_detected:
            print(f"ERROR: NaN at step {step}")
            survival_time = step * dt
            break

        if ball_off_plate(data):
            survival_time = (step + 1) * dt
            break

        # Periodic diagnostics
        t = (step + 1) * dt
        if step % 200 == 0:
            print(f"  t={t:.1f}s  error: x={error_x:+.4f} y={error_y:+.4f}  "
                  f"ctrl6={data.ctrl[5]:.3f} ctrl7={data.ctrl[6]:.3f}")

        # Render
        if step % render_every == 0:
            renderer.update_scene(data, camera=cam_id)
            frames.append(renderer.render())

    print()
    print(f"Survival Time: {survival_time:.1f} seconds")

    mediapy.write_video("attempt_1.mp4", frames, fps=fps)
    print(f"Video saved: attempt_1.mp4 ({len(frames)} frames)")

# ============================================================
# Streaming mode: live MJPEG with auto-reset loop
# ============================================================
else:
    print(f"Starting live stream on port {args.port}...")
    print(f"PID gains: Kp={KP}, Ki={KI}, Kd={KD}")
    print(f"Press Ctrl+C to stop.\n")

    streamer = LiveStreamer(port=args.port)
    streamer.start()
    fps = 30
    render_every = int(1.0 / (fps * dt))
    attempt = 0

    try:
        while True:
            # Reset for new attempt
            attempt += 1
            reset_scene(model, data)
            run_joint_diagnostics(model, data)

            pid_x = PIDController(KP, KI, KD, dt)
            pid_y = PIDController(KP, KI, KD, dt)

            print(f"--- Attempt {attempt} ---")
            step = 0
            survival_time = 0.0

            while True:
                error_x, error_y, nan_detected = run_simulation_step(data, pid_x, pid_y)
                step += 1

                if nan_detected:
                    print(f"ERROR: NaN at step {step}")
                    survival_time = step * dt
                    break

                if ball_off_plate(data):
                    survival_time = (step) * dt
                    break

                # Periodic diagnostics
                t = step * dt
                if step % 200 == 0:
                    print(f"  t={t:.1f}s  error: x={error_x:+.4f} y={error_y:+.4f}  "
                          f"ctrl6={data.ctrl[5]:.3f} ctrl7={data.ctrl[6]:.3f}")

                # Stream frame
                if step % render_every == 0:
                    renderer.update_scene(data, camera=cam_id)
                    streamer.update(renderer.render())

            print(f"Survival Time: {survival_time:.1f} seconds")

            # Let ball fall to the floor before resetting
            if not nan_detected:
                max_fall = int(3.0 / dt)
                settle = int(0.5 / dt)
                landed_at = None
                for fall_step in range(max_fall):
                    mujoco.mj_step(model, data)
                    for i in range(7):
                        data.ctrl[i] = home[i]
                    data.ctrl[7] = 0.008
                    if fall_step % render_every == 0:
                        renderer.update_scene(data, camera=cam_id)
                        streamer.update(renderer.render())
                    ball_z = data.xpos[ball_id][2]
                    if landed_at is None and ball_z < 0.05:
                        landed_at = fall_step
                    if landed_at is not None and (fall_step - landed_at) >= settle:
                        break
                    if np.any(np.isnan(data.xpos[ball_id])):
                        break
            print()

    except KeyboardInterrupt:
        print("\nStreaming stopped.")
    finally:
        streamer.stop()

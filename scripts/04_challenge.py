"""Sprint 3: Progressive disturbance challenge for ball-on-plate balancing.

4 levels of increasing difficulty to test PID robustness:
  Level 1 — Static Hold (no disturbance, 10s survival)
  Level 2 — Periodic Impulses (random forces every 2s)
  Level 3 — Moving Target (sinusoidal joint offsets)
  Level 4 — Speed Record (increasing frequency until ball falls)

Uses CORRECT joints and sign: joint6 (ctrl[5]) for X, joint7 (ctrl[6]) for Y, POSITIVE sign.

Run with: python scripts/04_challenge.py --level 2 --kp 2 --kd 0 [--force 1.0] [--freq 0.5]
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
parser = argparse.ArgumentParser(description="Progressive disturbance challenge")
parser.add_argument("--level", type=int, default=1, choices=[1, 2, 3, 4],
                    help="Challenge level 1-4 (default: 1)")
parser.add_argument("--kp", type=float, default=2.0,
                    help="Proportional gain (default: 2.0)")
parser.add_argument("--kd", type=float, default=0.0,
                    help="Derivative gain (default: 0.0)")
parser.add_argument("--force", type=float, default=1.0,
                    help="Impulse force magnitude for Level 2 (default: 1.0N)")
parser.add_argument("--freq", type=float, default=0.5,
                    help="Oscillation frequency for Level 3/4 (default: 0.5 Hz)")
parser.add_argument("--port", type=int, default=18080,
                    help="Streaming port (default: 18080)")
parser.add_argument("--no-stream", action="store_true",
                    help="Disable live streaming; save .mp4 instead")
args = parser.parse_args()

KP = args.kp
KI = 0.0
KD = args.kd

# --- Setup ---
model = mujoco.MjModel.from_xml_path("content/panda_ball_balance.xml")
data = mujoco.MjData(model)
dt = model.opt.timestep  # 0.005s = 200 Hz

plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")

# Home pose
home = [0.0, -0.785, 0.0, -2.356, 1.184, 3.184, 1.158]
joint_names = [f"joint{i}" for i in range(1, 8)]

# Joint IDs (cached)
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


def ball_off_plate(d):
    """Check if ball has fallen off the plate."""
    ball_rel_world = d.xpos[ball_id] - d.xpos[plate_id]
    error_x = ball_rel_world[0]
    error_y = ball_rel_world[1]
    return (abs(error_x) > 0.14 or abs(error_y) > 0.14 or ball_rel_world[2] < -0.02)


def level_header(level):
    """Print the level header with parameters."""
    if level == 1:
        print(f"=== Level 1: Static Hold ===")
    elif level == 2:
        print(f"=== Level 2: Periodic Impulses (force={args.force}N) ===")
    elif level == 3:
        print(f"=== Level 3: Moving Target (freq={args.freq} Hz) ===")
    elif level == 4:
        print(f"=== Level 4: Speed Record (start freq={args.freq} Hz) ===")
    print(f"PID: Kp={KP}, Kd={KD} (joint6+joint7, sign=+)")


def run_one_attempt(level, rng, max_duration=None):
    """Run a single attempt. Returns survival_time and frames list (if rendering).

    Parameters
    ----------
    level : int
        Challenge level (1-4).
    rng : np.random.RandomState
        Random state for reproducibility.
    max_duration : float or None
        Max duration in seconds. None means 10s for levels 1-3,
        unlimited for level 4.
    """
    reset_scene(model, data)

    pid_x = PIDController(KP, KI, KD, dt)
    pid_y = PIDController(KP, KI, KD, dt)

    if max_duration is None:
        if level == 4:
            max_duration = 600.0  # 10 minutes max for speed record
        else:
            max_duration = 10.0

    steps = int(max_duration / dt)
    fps = 30
    render_every = int(1.0 / (fps * dt))
    frames = []
    survival_time = max_duration

    # Level 2: impulse tracking
    impulse_interval = 2.0  # seconds
    impulse_duration_steps = 10  # 0.05s at 200 Hz
    impulse_active = False
    impulse_step_count = 0
    next_impulse_time = impulse_interval

    # Level 3/4: oscillation parameters
    amplitude = 0.02  # radians
    current_freq = args.freq

    # Level 4: frequency ramp
    freq_increase = 0.1  # Hz per 10 seconds
    max_freq_achieved = current_freq

    for step in range(steps):
        t = (step + 1) * dt

        # --- Level 2: Periodic impulse forces ---
        if level == 2:
            if impulse_active:
                impulse_step_count += 1
                if impulse_step_count >= impulse_duration_steps:
                    # Clear force
                    data.xfrc_applied[ball_id, :3] = 0
                    impulse_active = False
            elif t >= next_impulse_time:
                # Apply random force
                force_dir = rng.randn(3)
                force_dir = force_dir / np.linalg.norm(force_dir) * args.force
                data.xfrc_applied[ball_id, :3] = force_dir
                impulse_active = True
                impulse_step_count = 0
                next_impulse_time += impulse_interval
                print(f"  [t={t:.1f}s] IMPULSE: force=({force_dir[0]:+.2f}, "
                      f"{force_dir[1]:+.2f}, {force_dir[2]:+.2f})N")

        # --- Level 4: frequency ramp ---
        if level == 4:
            current_freq = args.freq + freq_increase * (t // 10.0)
            if current_freq > max_freq_achieved:
                max_freq_achieved = current_freq

        # Step simulation
        mujoco.mj_step(model, data)

        # NaN guard
        if np.any(np.isnan(data.xpos[ball_id])):
            print(f"ERROR: NaN at step {step}")
            survival_time = step * dt
            break

        # Hold non-PID joints at home
        for i in [0, 1, 2, 3, 4]:  # joint1-5
            data.ctrl[i] = home[i]
        data.ctrl[7] = 0.008  # gripper

        # Sense: ball position relative to plate
        ball_rel_world = data.xpos[ball_id] - data.xpos[plate_id]
        error_x = ball_rel_world[0]
        error_y = ball_rel_world[1]

        # PID corrections
        correction_x = pid_x.compute(error_x)
        correction_y = pid_y.compute(error_y)

        # Apply corrections: joint6 (ctrl[5]) for X, joint7 (ctrl[6]) for Y
        # POSITIVE sign (correct solution)
        if level in (3, 4):
            freq = current_freq if level == 4 else args.freq
            data.ctrl[5] = home[5] + correction_x + amplitude * np.cos(2 * np.pi * freq * t)
            data.ctrl[6] = home[6] + correction_y + amplitude * np.sin(2 * np.pi * freq * t)
        else:
            data.ctrl[5] = home[5] + correction_x
            data.ctrl[6] = home[6] + correction_y

        # Check ball off plate
        if ball_off_plate(data):
            survival_time = t
            break

        # Periodic status
        if step % 400 == 0 and step > 0:
            status = f"  t={t:.1f}s  error: x={error_x:+.4f} y={error_y:+.4f}"
            if level == 4:
                status += f"  freq={current_freq:.1f} Hz"
            print(status)

        # Render
        if step % render_every == 0:
            renderer.update_scene(data, camera=cam_id)
            frames.append(renderer.render())

    # Level 2: clear any residual force
    if level == 2:
        data.xfrc_applied[ball_id, :3] = 0

    # Print results
    if level == 4:
        if survival_time >= max_duration:
            print(f"Survived full {max_duration:.0f}s! Max frequency: {max_freq_achieved:.1f} Hz")
        else:
            print(f"Max frequency achieved: {max_freq_achieved:.1f} Hz")

    print(f"Survival Time: {survival_time:.1f} seconds")

    return survival_time, frames


# ============================================================
# .mp4 mode: single run, save video, exit
# ============================================================
if args.no_stream or not HAS_STREAMER:
    import mediapy

    if not args.no_stream and not HAS_STREAMER:
        print("WARNING: mujoco_streamer not installed, falling back to .mp4 output")

    rng = np.random.RandomState(42)
    level_header(args.level)
    survival_time, frames = run_one_attempt(args.level, rng)
    print()

    filename = f"challenge_level{args.level}.mp4"
    mediapy.write_video(filename, frames, fps=30)
    print(f"Video saved: {filename} ({len(frames)} frames)")

# ============================================================
# Streaming mode: live MJPEG with auto-reset loop
# ============================================================
else:
    print(f"Starting live stream on port {args.port}...")
    level_header(args.level)
    print(f"Press Ctrl+C to stop.\n")

    streamer = LiveStreamer(port=args.port)
    fps = 30
    render_every = int(1.0 / (fps * dt))
    attempt = 0

    try:
        while True:
            attempt += 1
            rng = np.random.RandomState(42 + attempt)
            print(f"--- Attempt {attempt} ---")

            reset_scene(model, data)
            pid_x = PIDController(KP, KI, KD, dt)
            pid_y = PIDController(KP, KI, KD, dt)

            # Level 2: impulse tracking
            impulse_interval = 2.0
            impulse_duration_steps = 10
            impulse_active = False
            impulse_step_count = 0
            next_impulse_time = impulse_interval

            # Level 3/4: oscillation
            amplitude = 0.02
            current_freq = args.freq
            freq_increase = 0.1
            max_freq_achieved = current_freq

            step = 0
            survival_time = 0.0
            max_duration = 600.0 if args.level == 4 else 10.0

            while True:
                step += 1
                t = step * dt

                if t > max_duration:
                    survival_time = max_duration
                    break

                # --- Level 2: Periodic impulse forces ---
                if args.level == 2:
                    if impulse_active:
                        impulse_step_count += 1
                        if impulse_step_count >= impulse_duration_steps:
                            data.xfrc_applied[ball_id, :3] = 0
                            impulse_active = False
                    elif t >= next_impulse_time:
                        force_dir = rng.randn(3)
                        force_dir = force_dir / np.linalg.norm(force_dir) * args.force
                        data.xfrc_applied[ball_id, :3] = force_dir
                        impulse_active = True
                        impulse_step_count = 0
                        next_impulse_time += impulse_interval
                        print(f"  [t={t:.1f}s] IMPULSE: force=({force_dir[0]:+.2f}, "
                              f"{force_dir[1]:+.2f}, {force_dir[2]:+.2f})N")

                # --- Level 4: frequency ramp ---
                if args.level == 4:
                    current_freq = args.freq + freq_increase * (t // 10.0)
                    if current_freq > max_freq_achieved:
                        max_freq_achieved = current_freq

                # Step simulation
                mujoco.mj_step(model, data)

                # NaN guard
                if np.any(np.isnan(data.xpos[ball_id])):
                    print(f"ERROR: NaN at step {step}")
                    survival_time = step * dt
                    break

                # Hold non-PID joints at home
                for i in [0, 1, 2, 3, 4]:
                    data.ctrl[i] = home[i]
                data.ctrl[7] = 0.008

                # Sense
                ball_rel_world = data.xpos[ball_id] - data.xpos[plate_id]
                error_x = ball_rel_world[0]
                error_y = ball_rel_world[1]

                # PID
                correction_x = pid_x.compute(error_x)
                correction_y = pid_y.compute(error_y)

                # Apply: joint6 (ctrl[5]) for X, joint7 (ctrl[6]) for Y, POSITIVE sign
                if args.level in (3, 4):
                    freq = current_freq if args.level == 4 else args.freq
                    data.ctrl[5] = home[5] + correction_x + amplitude * np.cos(2 * np.pi * freq * t)
                    data.ctrl[6] = home[6] + correction_y + amplitude * np.sin(2 * np.pi * freq * t)
                else:
                    data.ctrl[5] = home[5] + correction_x
                    data.ctrl[6] = home[6] + correction_y

                # Check ball off plate
                if ball_off_plate(data):
                    survival_time = t
                    break

                # Periodic status
                if step % 400 == 0:
                    status = f"  t={t:.1f}s  error: x={error_x:+.4f} y={error_y:+.4f}"
                    if args.level == 4:
                        status += f"  freq={current_freq:.1f} Hz"
                    print(status)

                # Stream frame
                if step % render_every == 0:
                    renderer.update_scene(data, camera=cam_id)
                    streamer.update(renderer.render())

            # Clean up forces
            if args.level == 2:
                data.xfrc_applied[ball_id, :3] = 0

            # Print results
            if args.level == 4:
                if survival_time >= max_duration:
                    print(f"Survived full {max_duration:.0f}s! Max frequency: {max_freq_achieved:.1f} Hz")
                else:
                    print(f"Max frequency achieved: {max_freq_achieved:.1f} Hz")

            print(f"Survival Time: {survival_time:.1f} seconds")
            print()

    except KeyboardInterrupt:
        print("\nStreaming stopped.")
    finally:
        streamer.stop()

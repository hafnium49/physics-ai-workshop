"""Step 4: Validate that a 10-second solution exists.

Systematically tests joint pairings and PID sign/gain combinations.
Proves the task is solvable and identifies the correct control architecture.

Default:  python scripts/03_optimize_pid.py               (grid search + render best as stream)
Fallback: python scripts/03_optimize_pid.py --no-stream   (grid search + render best as .mp4)
Dry run:  python scripts/03_optimize_pid.py --no-render   (grid search only, no video output)
"""
import os
import sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# Optional streamer import
# ---------------------------------------------------------------------------
try:
    from mujoco_streamer import LiveStreamer
    HAS_STREAMER = True
except ImportError:
    HAS_STREAMER = False

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="PID optimization grid search for ball-on-plate balancing")
parser.add_argument("--no-stream", action="store_true",
                    help="Disable live streaming; save .mp4 instead")
parser.add_argument("--no-render", action="store_true",
                    help="Skip all rendering (dry run, no video output)")
parser.add_argument("--port", type=int, default=18080,
                    help="MJPEG streaming port (default: 18080)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model = mujoco.MjModel.from_xml_path("content/panda_ball_balance.xml")

plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
ball_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")

home = [0.0, -0.785, 0.0, -2.356, 0.0, 1.8, 0.785]
joint_names = [f"joint{i}" for i in range(1, 8)]


def run_trial(joint_x_idx, joint_y_idx, sign, kp, kd, duration=10.0, render=False):
    """Run a simulation trial with given joint pairing, sign, and PID gains.

    Args:
        joint_x_idx: actuator index for X-axis control (0-based)
        joint_y_idx: actuator index for Y-axis control (0-based)
        sign: +1 or -1 for correction direction
        kp, kd: PID gains
        duration: simulation length in seconds
        render: if True, return (survival_time, frames)

    Returns:
        survival_time (float), or (survival_time, frames) if render=True
    """
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    # Set arm to home pose
    for jn, val in zip(joint_names, home):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        data.qpos[model.jnt_qposadr[jid]] = val
    for i, val in enumerate(home):
        data.ctrl[i] = val
    data.ctrl[7] = 0.008
    mujoco.mj_forward(model, data)

    # Place ball on plate
    ba = model.jnt_qposadr[ball_joint_id]
    bv = model.jnt_dofadr[ball_joint_id]
    data.qpos[ba:ba + 3] = data.xpos[plate_id] + [0, 0, 0.025]
    data.qpos[ba + 3:ba + 7] = [1, 0, 0, 0]
    data.qvel[bv:bv + 6] = 0
    mujoco.mj_forward(model, data)

    renderer = None
    frames = []
    cam_id = -1
    if render:
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side")

    fps = 30
    render_every = int(1.0 / (fps * dt))
    prev_ex, prev_ey = 0.0, 0.0
    steps = int(duration / dt)

    for step in range(steps):
        mujoco.mj_step(model, data)

        # Hold all joints at home
        for i, val in enumerate(home):
            data.ctrl[i] = val
        data.ctrl[7] = 0.008

        # Ball error
        brel = data.xpos[ball_id] - data.xpos[plate_id]
        ex, ey = brel[0], brel[1]
        dx = (ex - prev_ex) / dt
        dy = (ey - prev_ey) / dt

        # Apply PID correction to selected joints
        data.ctrl[joint_y_idx] = home[joint_y_idx] + sign * (kp * ey + kd * dy)
        data.ctrl[joint_x_idx] = home[joint_x_idx] + sign * (kp * ex + kd * dx)

        prev_ex, prev_ey = ex, ey

        # NaN check
        if np.any(np.isnan(data.xpos[ball_id])):
            t = step * dt
            return (t, frames) if render else t

        # Ball off plate
        if abs(ex) > 0.14 or abs(ey) > 0.14 or brel[2] < -0.02:
            t = (step + 1) * dt
            return (t, frames) if render else t

        # Render
        if render and step % render_every == 0:
            renderer.update_scene(data, camera=cam_id)
            frames.append(renderer.render())

    return (duration, frames) if render else duration


# --- Phase 1: Find the correct joint pairing ---
print("=" * 60)
print("Phase 1: Testing joint pairings (Kp=50, Kd=10)")
print("=" * 60)

# Pairings: (name, jx_ctrl_idx, jy_ctrl_idx)
# jx = actuator index controlling plate X, jy = actuator index controlling plate Y
# Empirically: joint6 (ctrl[5]) -> plate X, joint5 (ctrl[4]) -> plate Y
pairings = [
    ("j6(X)+j5(Y)", 5, 4),   # correct pairing
    ("j6(X)+j7(Y)", 5, 6),   # j7 has no Y authority
    ("j5(X)+j6(Y)", 4, 5),   # axes swapped
    ("j5(X)+j4(Y)", 4, 3),   # wrong joints entirely
]

for name, jx, jy in pairings:
    for sign in [+1, -1]:
        t = run_trial(jx, jy, sign, kp=50, kd=10, duration=5.0)
        marker = " <-- WORKS" if t >= 5.0 else ""
        print(f"  {name} sign={sign:+d} -> {t:.1f}s{marker}")

# --- Phase 2: Confirm with the winning pairing ---
print()
print("=" * 60)
print("Phase 2: Gain search with j6(X)+j5(Y), sign=+1")
print("=" * 60)

results = []
for kp in [5, 10, 20, 50, 100]:
    for kd in [1, 5, 10, 20]:
        t = run_trial(5, 4, +1, kp, kd)
        results.append((kp, kd, t))
        marker = " ***" if t >= 10.0 else ""
        print(f"  Kp={kp:>3d} Kd={kd:>2d} -> Survival Time: {t:.1f}s{marker}")

best = max(results, key=lambda x: x[2])
print(f"\nBest: Kp={best[0]}, Kd={best[1]}, Survival={best[2]:.1f}s")

# --- Phase 3: Render the best result (unless --no-render) ---
if args.no_render:
    print("\n--no-render specified, skipping video output.")
else:
    use_stream = (not args.no_stream) and HAS_STREAMER

    if use_stream:
        # ---- Live MJPEG streaming of best result ----
        print(f"\nStreaming best result on port {args.port}...")
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side")
        streamer = LiveStreamer(port=args.port)
        streamer.start()

        # Run the best trial and stream frames
        data = mujoco.MjData(model)
        dt = model.opt.timestep

        # Reset scene
        for jn, val in zip(joint_names, home):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            data.qpos[model.jnt_qposadr[jid]] = val
        for i, val in enumerate(home):
            data.ctrl[i] = val
        data.ctrl[7] = 0.008
        mujoco.mj_forward(model, data)

        ba = model.jnt_qposadr[ball_joint_id]
        bv = model.jnt_dofadr[ball_joint_id]
        data.qpos[ba:ba + 3] = data.xpos[plate_id] + [0, 0, 0.025]
        data.qpos[ba + 3:ba + 7] = [1, 0, 0, 0]
        data.qvel[bv:bv + 6] = 0
        mujoco.mj_forward(model, data)

        fps = 30
        render_every = int(1.0 / (fps * dt))
        prev_ex, prev_ey = 0.0, 0.0
        steps = int(10.0 / dt)

        print(f"Streaming Kp={best[0]}, Kd={best[1]} for 10s...")
        print("Press Ctrl+C to stop.\n")

        try:
            for step in range(steps):
                mujoco.mj_step(model, data)

                for i, val in enumerate(home):
                    data.ctrl[i] = val
                data.ctrl[7] = 0.008

                brel = data.xpos[ball_id] - data.xpos[plate_id]
                ex, ey = brel[0], brel[1]
                dx = (ex - prev_ex) / dt
                dy = (ey - prev_ey) / dt

                # j6(X)+j5(Y) with positive sign
                data.ctrl[5] = home[5] + (best[0] * ex + best[1] * dx)  # joint6 for X
                data.ctrl[4] = home[4] + (best[0] * ey + best[1] * dy)  # joint5 for Y

                prev_ex, prev_ey = ex, ey

                if np.any(np.isnan(data.xpos[ball_id])):
                    break
                if abs(ex) > 0.14 or abs(ey) > 0.14 or brel[2] < -0.02:
                    break

                if step % render_every == 0:
                    renderer.update_scene(data, camera=cam_id)
                    streamer.update(renderer.render())
        except KeyboardInterrupt:
            print("\nStreaming stopped.")
        finally:
            streamer.stop()

    else:
        # ---- .mp4 fallback mode ----
        import mediapy

        if not args.no_stream and not HAS_STREAMER:
            print("WARNING: mujoco_streamer not installed, falling back to .mp4 output")

        print("\nRendering best result...")
        t, frames = run_trial(4, 5, +1, best[0], best[1], render=True)
        mediapy.write_video("best_balance.mp4", frames, fps=30)
        print(f"Video saved: best_balance.mp4 ({len(frames)} frames, {t:.1f}s survival)")

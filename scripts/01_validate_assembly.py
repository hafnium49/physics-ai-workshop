"""Step 2: Validate the combined panda_ball_balance.xml model.

Loads the model, sets the arm to home pose, places the ball on the plate,
and either live-streams MJPEG or saves a short .mp4.

Default:  MUJOCO_GL=egl python scripts/01_validate_assembly.py        (live stream)
Fallback: MUJOCO_GL=egl python scripts/01_validate_assembly.py --no-stream --duration 3
"""
import os
import sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

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
parser = argparse.ArgumentParser(description="Validate panda + ball-on-plate assembly")
parser.add_argument("--no-stream", action="store_true",
                    help="Disable live streaming; save .mp4 instead")
parser.add_argument("--port", type=int, default=8080,
                    help="MJPEG streaming port (default: 8080)")
parser.add_argument("--duration", type=float, default=3.0,
                    help="Video duration in seconds for .mp4 mode (default: 3.0)")
args = parser.parse_args()

use_stream = (not args.no_stream) and HAS_STREAMER

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model = mujoco.MjModel.from_xml_path("content/panda_ball_balance.xml")
data = mujoco.MjData(model)

# Body IDs
plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")

# Joint IDs and qpos addresses
joint_names = [f"joint{i}" for i in range(1, 8)]
joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
joint_addrs = [model.jnt_qposadr[jid] for jid in joint_ids]

# Home pose: j6=1.8 (mid-range) to address asymmetric range [-0.0175, 3.7525]
home = [0.0, -0.785, 0.0, -2.356, 0.0, 1.8, 0.785]

# Set arm to home pose
for addr, val in zip(joint_addrs, home):
    data.qpos[addr] = val

# Set actuator controls to hold home pose
for i, val in enumerate(home):
    data.ctrl[i] = val
# Close gripper
data.ctrl[7] = 0.04

# Forward pass to compute plate world position
mujoco.mj_forward(model, data)

# Place ball on plate: plate position + 0.025m above surface
plate_pos = data.xpos[plate_id].copy()
ball_qpos_addr = model.jnt_qposadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
]
# Free joint qpos: [x, y, z, qw, qx, qy, qz]
data.qpos[ball_qpos_addr:ball_qpos_addr + 3] = plate_pos + [0, 0, 0.025]
data.qpos[ball_qpos_addr + 3:ball_qpos_addr + 7] = [1, 0, 0, 0]  # identity quat
# Zero ball velocity
ball_qvel_addr = model.jnt_dofadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
]
data.qvel[ball_qvel_addr:ball_qvel_addr + 6] = 0

mujoco.mj_forward(model, data)

print(f"Plate world pos: {data.xpos[plate_id]}")
print(f"Ball world pos:  {data.xpos[ball_id]}")
print(f"Ball-plate Z gap: {data.xpos[ball_id][2] - data.xpos[plate_id][2]:.4f} m")

# ---------------------------------------------------------------------------
# Renderer (side camera)
# ---------------------------------------------------------------------------
renderer = mujoco.Renderer(model, height=480, width=640)
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side")

fps = 30
render_every = int(1.0 / (fps * model.opt.timestep))


def ball_on_plate_check():
    """Return True if the ball is roughly above the plate."""
    rel = data.xpos[ball_id] - data.xpos[plate_id]
    return abs(rel[0]) < 0.15 and abs(rel[1]) < 0.15 and rel[2] > -0.02


def print_diagnostics(step):
    """Print plate/ball positions and ball-on-plate status."""
    ball_rel = data.xpos[ball_id] - data.xpos[plate_id]
    print(f"[t={data.time:.1f}s] "
          f"plate={data.xpos[plate_id]}  "
          f"ball={data.xpos[ball_id]}  "
          f"rel=({ball_rel[0]:+.4f}, {ball_rel[1]:+.4f}, {ball_rel[2]:+.4f})  "
          f"on_plate={ball_on_plate_check()}")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
if use_stream:
    # ---- Live MJPEG streaming mode ----
    streamer = LiveStreamer(port=args.port)
    streamer.start()
    print(f"\nLive stream started on http://localhost:{args.port}")
    print("Press Ctrl+C to stop.\n")

    step = 0
    diag_interval = int(1.0 / model.opt.timestep)  # every ~1 second
    try:
        while True:
            mujoco.mj_step(model, data)

            # Hold arm at home
            for i, val in enumerate(home):
                data.ctrl[i] = val

            # NaN check
            if np.any(np.isnan(data.xpos[ball_id])):
                print(f"ERROR: NaN detected at step {step}")
                break

            # Render & push frame
            if step % render_every == 0:
                renderer.update_scene(data, camera=cam_id)
                streamer.update(renderer.render())

            # Diagnostics every second
            if step % diag_interval == 0:
                print_diagnostics(step)

            step += 1
    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        streamer.stop()
        print("Stream stopped.")

else:
    # ---- .mp4 fallback mode ----
    import mediapy

    if not args.no_stream and not HAS_STREAMER:
        print("WARNING: mujoco_streamer not installed. Falling back to .mp4 output.\n")

    duration = args.duration
    steps = int(duration / model.opt.timestep)
    diag_interval = int(1.0 / model.opt.timestep)  # every ~1 second
    frames = []

    for step in range(steps):
        mujoco.mj_step(model, data)

        # Hold arm at home
        for i, val in enumerate(home):
            data.ctrl[i] = val

        # NaN check
        if np.any(np.isnan(data.xpos[ball_id])):
            print(f"ERROR: NaN detected at step {step}")
            break

        # Render frame
        if step % render_every == 0:
            renderer.update_scene(data, camera=cam_id)
            frames.append(renderer.render())

        # Diagnostics every second
        if step % diag_interval == 0:
            print_diagnostics(step)

    # Final diagnostics
    ball_rel = data.xpos[ball_id] - data.xpos[plate_id]
    print(f"\nFinal plate pos: {data.xpos[plate_id]}")
    print(f"Final ball pos:  {data.xpos[ball_id]}")
    print(f"Ball relative to plate: x={ball_rel[0]:.4f} y={ball_rel[1]:.4f} z={ball_rel[2]:.4f}")
    print(f"Ball on plate: {ball_on_plate_check()}")

    mediapy.write_video("assembly_test.mp4", frames, fps=fps)
    print(f"\nVideo saved: assembly_test.mp4 ({len(frames)} frames)")

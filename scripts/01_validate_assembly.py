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
parser.add_argument("--port", type=int, default=18080,
                    help="MJPEG streaming port (default: 18080)")
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
hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

# Joint IDs and qpos addresses
joint_names = [f"joint{i}" for i in range(1, 8)]
joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
joint_addrs = [model.jnt_qposadr[jid] for jid in joint_ids]

# Plate free joint addresses
plate_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "plate_free")
plate_qpos_addr = model.jnt_qposadr[plate_joint_id]
plate_qvel_addr = model.jnt_dofadr[plate_joint_id]

# Ball free joint addresses
ball_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
ball_qpos_addr = model.jnt_qposadr[ball_joint_id]
ball_qvel_addr = model.jnt_dofadr[ball_joint_id]

# Home pose: j6=1.8 (mid-range) to address asymmetric range [-0.0175, 3.7525]
home = [0.0, -0.785, 0.0, -2.356, 0.0, 1.8, 0.785]

# ---------------------------------------------------------------------------
# Phase 1: Set arm to home pose with gripper open
# ---------------------------------------------------------------------------
for addr, val in zip(joint_addrs, home):
    data.qpos[addr] = val
for i, val in enumerate(home):
    data.ctrl[i] = val
data.ctrl[7] = 0.04  # Open gripper
mujoco.mj_forward(model, data)

# ---------------------------------------------------------------------------
# Phase 2: Position plate edge between gripper fingers (horizontal plate)
# ---------------------------------------------------------------------------
hand_pos = data.xpos[hand_id].copy()
hand_mat = data.xmat[hand_id].reshape(3, 3)
hand_quat = data.xquat[hand_id].copy()

# Finger pads are at approximately (0, 0, 0.104) in hand frame
grip_local = np.array([0.0, 0.0, 0.104])
grip_world = hand_pos + hand_mat @ grip_local

# Hand approach direction projected to horizontal (plate extends this way)
hand_approach = hand_mat[:, 2].copy()  # hand Z = approach direction
hand_approach_horiz = hand_approach.copy()
hand_approach_horiz[2] = 0.0
hand_approach_horiz /= np.linalg.norm(hand_approach_horiz)

# Plate is horizontal, center offset from grip by half-width along approach dir
plate_center = grip_world + hand_approach_horiz * 0.15
plate_center[2] = grip_world[2]  # keep at grip height
plate_quat = np.array([1.0, 0.0, 0.0, 0.0])  # horizontal (identity)

# Set plate free joint
data.qpos[plate_qpos_addr:plate_qpos_addr + 3] = plate_center
data.qpos[plate_qpos_addr + 3:plate_qpos_addr + 7] = plate_quat
data.qvel[plate_qvel_addr:plate_qvel_addr + 6] = 0
mujoco.mj_forward(model, data)

# ---------------------------------------------------------------------------
# Phase 3: Update weld constraint relpose, then close gripper and settle
# ---------------------------------------------------------------------------
# Compute desired relative pose of plate in hand frame
rel_pos = hand_mat.T @ (plate_center - hand_pos)
hand_quat_inv = np.zeros(4)
mujoco.mju_negQuat(hand_quat_inv, hand_quat)
rel_quat = np.zeros(4)
mujoco.mju_mulQuat(rel_quat, hand_quat_inv, plate_quat)

# Find and update the weld constraint (eq_type 1 = mjEQ_WELD)
for i in range(model.neq):
    if model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
        model.eq_data[i, 3:6] = rel_pos
        model.eq_data[i, 6:10] = rel_quat
        break

data.ctrl[7] = 0.0  # Close gripper around plate edge
print("Closing gripper on plate edge...")
print(f"  Grip point (world): {grip_world}")
print(f"  Plate center: {plate_center}")

for _ in range(400):  # 2 seconds settling at 200 Hz
    mujoco.mj_step(model, data)

print(f"  Plate position after grip: {data.xpos[plate_id]}")

# ---------------------------------------------------------------------------
# Phase 4: Place ball on plate
# ---------------------------------------------------------------------------
plate_pos = data.xpos[plate_id].copy()
data.qpos[ball_qpos_addr:ball_qpos_addr + 3] = plate_pos + [0, 0, 0.025]
data.qpos[ball_qpos_addr + 3:ball_qpos_addr + 7] = [1, 0, 0, 0]
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

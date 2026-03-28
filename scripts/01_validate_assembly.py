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

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)

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
parser.add_argument("--port", type=int, default=None,
                    help="MJPEG streaming port (default: STREAM_PORT env or 18080)")
parser.add_argument("--duration", type=float, default=3.0,
                    help="Video duration in seconds for .mp4 mode (default: 3.0)")
args = parser.parse_args()

stream_port = args.port if args.port is not None else int(os.environ.get("STREAM_PORT", 18080))

use_stream = (not args.no_stream) and HAS_STREAMER

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model = mujoco.MjModel.from_xml_path(os.path.join(_project_root, "content", "panda_ball_balance.xml"))
data = mujoco.MjData(model)

# Body IDs
plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")

# Joint IDs and qpos addresses
joint_names = [f"joint{i}" for i in range(1, 8)]
joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
joint_addrs = [model.jnt_qposadr[jid] for jid in joint_ids]

# Home pose: j5, j6, j7 adjusted for plate grip orientation
home = [0.0, -0.785, 0.0, -2.356, 1.184, 3.184, 1.158]

ball_qpos_addr = model.jnt_qposadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
]
ball_qvel_addr = model.jnt_dofadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
]


def reset_scene():
    """Set arm to home pose, place ball on plate, zero velocities."""
    mujoco.mj_resetData(model, data)
    for addr, val in zip(joint_addrs, home):
        data.qpos[addr] = val
    for i, val in enumerate(home):
        data.ctrl[i] = val
    data.ctrl[7] = 0.008
    mujoco.mj_forward(model, data)

    plate_pos = data.xpos[plate_id].copy()
    data.qpos[ball_qpos_addr:ball_qpos_addr + 3] = plate_pos + [0, 0, 0.025]
    data.qpos[ball_qpos_addr + 3:ball_qpos_addr + 7] = [1, 0, 0, 0]
    data.qvel[ball_qvel_addr:ball_qvel_addr + 6] = 0
    mujoco.mj_forward(model, data)


reset_scene()

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
    streamer = LiveStreamer(port=stream_port)
    streamer.start()
    cam = streamer.make_free_camera(model)
    print(f"\nLive stream started on http://localhost:{stream_port}")
    print("Press Ctrl+C to stop.\n")

    step = 0
    diag_interval = int(1.0 / model.opt.timestep)  # every ~1 second
    try:
        while True:  # outer loop: recovery on NaN
            while True:  # inner loop: simulation steps
                mujoco.mj_step(model, data)

                # Hold arm at home
                for i, val in enumerate(home):
                    data.ctrl[i] = val

                # NaN check — reset scene instead of exiting
                if np.any(np.isnan(data.xpos[ball_id])):
                    print(f"WARNING: NaN detected at step {step}, resetting scene...")
                    break

                # Render & push frame
                if step % render_every == 0:
                    streamer.drain_camera_commands(model, cam, renderer.scene)
                    renderer.update_scene(data, camera=cam)
                    streamer.update(renderer.render())

                # Diagnostics every second
                if step % diag_interval == 0:
                    print_diagnostics(step)

                step += 1

            # Recover from NaN: reset and continue streaming
            reset_scene()
            step = 0
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

    step = 0
    while step < steps:
        mujoco.mj_step(model, data)

        # Hold arm at home
        for i, val in enumerate(home):
            data.ctrl[i] = val

        # NaN check — reset scene instead of exiting
        if np.any(np.isnan(data.xpos[ball_id])):
            print(f"WARNING: NaN detected at step {step}, resetting scene...")
            reset_scene()
            step = 0
            frames.clear()
            continue

        # Render frame
        if step % render_every == 0:
            renderer.update_scene(data, camera=cam_id)
            frames.append(renderer.render())

        # Diagnostics every second
        if step % diag_interval == 0:
            print_diagnostics(step)

        step += 1

    # Final diagnostics
    ball_rel = data.xpos[ball_id] - data.xpos[plate_id]
    print(f"\nFinal plate pos: {data.xpos[plate_id]}")
    print(f"Final ball pos:  {data.xpos[ball_id]}")
    print(f"Ball relative to plate: x={ball_rel[0]:.4f} y={ball_rel[1]:.4f} z={ball_rel[2]:.4f}")
    print(f"Ball on plate: {ball_on_plate_check()}")

    mediapy.write_video("assembly_test.mp4", frames, fps=fps)
    print(f"\nVideo saved: assembly_test.mp4 ({len(frames)} frames)")

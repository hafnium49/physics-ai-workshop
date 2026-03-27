"""Step 2: Validate the combined panda_ball_balance.xml model.

Loads the model, sets the arm to home pose, places the ball on the plate,
runs a 3-second simulation, and renders to assembly_test.mp4.

Run with: MUJOCO_GL=egl python scripts/01_validate_assembly.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mediapy
import numpy as np

# Load model
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

# Renderer with fixed side camera
renderer = mujoco.Renderer(model, height=480, width=640)
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side")

# Simulation
duration = 3.0
fps = 30
steps = int(duration / model.opt.timestep)
render_every = int(1.0 / (fps * model.opt.timestep))
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

# Final diagnostics
ball_rel = data.xpos[ball_id] - data.xpos[plate_id]
print(f"\nFinal plate pos: {data.xpos[plate_id]}")
print(f"Final ball pos:  {data.xpos[ball_id]}")
print(f"Ball relative to plate: x={ball_rel[0]:.4f} y={ball_rel[1]:.4f} z={ball_rel[2]:.4f}")
ball_on_plate = abs(ball_rel[0]) < 0.15 and abs(ball_rel[1]) < 0.15 and ball_rel[2] > -0.02
print(f"Ball on plate: {ball_on_plate}")

mediapy.write_video("assembly_test.mp4", frames, fps=fps)
print(f"\nVideo saved: assembly_test.mp4 ({len(frames)} frames)")

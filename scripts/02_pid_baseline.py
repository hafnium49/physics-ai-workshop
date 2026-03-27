"""Step 3: Baseline PID controller for ball-on-plate balancing.

Applies a PID controller to joint6/joint7 to keep the ball centered on the plate.
Prints Survival Time to terminal. Deliberately mediocre gains for workshop demo.

Run with: MUJOCO_GL=egl python scripts/02_pid_baseline.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mediapy
import numpy as np


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


# --- PID gains (deliberately mediocre for baseline) ---
KP = 50.0
KI = 0.0
KD = 10.0

# --- Setup ---
model = mujoco.MjModel.from_xml_path("content/panda_ball_balance.xml")
data = mujoco.MjData(model)
dt = model.opt.timestep

plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")

# Home pose (j6=1.8 mid-range)
home = [0.0, -0.785, 0.0, -2.356, 0.0, 1.8, 0.785]
joint_names = [f"joint{i}" for i in range(1, 8)]

# Set arm to home
for jn, val in zip(joint_names, home):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
    data.qpos[model.jnt_qposadr[jid]] = val
for i, val in enumerate(home):
    data.ctrl[i] = val
data.ctrl[7] = 0.04  # close gripper

mujoco.mj_forward(model, data)

# Place ball on plate
ball_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
ball_qpos_addr = model.jnt_qposadr[ball_joint_id]
ball_qvel_addr = model.jnt_dofadr[ball_joint_id]
data.qpos[ball_qpos_addr:ball_qpos_addr + 3] = data.xpos[plate_id] + [0, 0, 0.025]
data.qpos[ball_qpos_addr + 3:ball_qpos_addr + 7] = [1, 0, 0, 0]
data.qvel[ball_qvel_addr:ball_qvel_addr + 6] = 0
mujoco.mj_forward(model, data)

# --- Hand rotation compensation ---
# The hand body has quat="0.9238795 0 0 -0.3826834" which is a -45 degree
# rotation around the X axis. We need to figure out the mapping between
# world-frame ball XY error and joint6/joint7 corrections empirically.
# For now, use the rotation matrix from the plate's world orientation.
plate_xmat = data.xmat[plate_id].reshape(3, 3)

# --- PID controllers for two axes ---
pid_a = PIDController(KP, KI, KD, dt)
pid_b = PIDController(KP, KI, KD, dt)

# --- Simulation ---
renderer = mujoco.Renderer(model, height=480, width=640)
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side")

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
    mujoco.mj_step(model, data)

    # Hold non-PID joints at home
    for i in range(5):  # joint1-5
        data.ctrl[i] = home[i]
    data.ctrl[7] = 0.04  # gripper

    # Sense: ball position relative to plate in world frame
    ball_rel_world = data.xpos[ball_id] - data.xpos[plate_id]

    # Transform to plate-local frame
    plate_xmat = data.xmat[plate_id].reshape(3, 3)
    ball_rel_local = plate_xmat.T @ ball_rel_world

    # PID on local X and Y errors
    correction_a = pid_a.compute(ball_rel_local[0])
    correction_b = pid_b.compute(ball_rel_local[1])

    # Apply corrections to joint6 and joint7
    data.ctrl[5] = home[5] - correction_a  # joint6
    data.ctrl[6] = home[6] - correction_b  # joint7

    # NaN check
    if np.any(np.isnan(data.xpos[ball_id])):
        print(f"ERROR: NaN at step {step}")
        survival_time = step * dt
        break

    # Check if ball fell off
    if abs(ball_rel_local[0]) > 0.14 or abs(ball_rel_local[1]) > 0.14 or ball_rel_world[2] < -0.02:
        survival_time = (step + 1) * dt
        break

    # Periodic diagnostics
    t = (step + 1) * dt
    if step % 200 == 0:
        print(f"  t={t:.1f}s  ball_local: x={ball_rel_local[0]:+.4f} y={ball_rel_local[1]:+.4f}  "
              f"ctrl6={data.ctrl[5]:.3f} ctrl7={data.ctrl[6]:.3f}")

    # Render
    if step % render_every == 0:
        renderer.update_scene(data, camera=cam_id)
        frames.append(renderer.render())

print()
print(f"Survival Time: {survival_time:.1f} seconds")

mediapy.write_video("attempt_1.mp4", frames, fps=fps)
print(f"Video saved: attempt_1.mp4 ({len(frames)} frames)")

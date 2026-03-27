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

# --- PID controllers: joint5 for Y, joint6 for X ---
# Empirically verified: joint6 +angle -> plate +X, joint5 +angle -> plate +Y
pid_x = PIDController(KP, KI, KD, dt)
pid_y = PIDController(KP, KI, KD, dt)

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
    for i in [0, 1, 2, 3]:  # joint1-4
        data.ctrl[i] = home[i]
    data.ctrl[6] = home[6]  # joint7 (no authority, hold)
    data.ctrl[7] = 0.04     # gripper

    # Sense: ball position relative to plate in world frame
    ball_rel_world = data.xpos[ball_id] - data.xpos[plate_id]

    # PID on world-frame X and Y errors
    # joint6 +angle -> plate moves +X, joint5 +angle -> plate moves +Y
    error_x = ball_rel_world[0]
    error_y = ball_rel_world[1]

    correction_x = pid_x.compute(error_x)
    correction_y = pid_y.compute(error_y)

    # Apply corrections: joint5 (ctrl[4]) for Y, joint6 (ctrl[5]) for X
    data.ctrl[4] = home[4] - correction_y  # joint5 controls plate Y
    data.ctrl[5] = home[5] - correction_x  # joint6 controls plate X

    # NaN check
    if np.any(np.isnan(data.xpos[ball_id])):
        print(f"ERROR: NaN at step {step}")
        survival_time = step * dt
        break

    # Check if ball fell off
    if abs(error_x) > 0.14 or abs(error_y) > 0.14 or ball_rel_world[2] < -0.02:
        survival_time = (step + 1) * dt
        break

    # Periodic diagnostics
    t = (step + 1) * dt
    if step % 200 == 0:
        print(f"  t={t:.1f}s  error: x={error_x:+.4f} y={error_y:+.4f}  "
              f"ctrl5={data.ctrl[4]:.3f} ctrl6={data.ctrl[5]:.3f}")

    # Render
    if step % render_every == 0:
        renderer.update_scene(data, camera=cam_id)
        frames.append(renderer.render())

print()
print(f"Survival Time: {survival_time:.1f} seconds")

mediapy.write_video("attempt_1.mp4", frames, fps=fps)
print(f"Video saved: attempt_1.mp4 ({len(frames)} frames)")

"""Controller Exploration Playground — Improve the ball-on-plate controller.

This file is your controller. Edit the make_controller() function to try
different control strategies, then measure your improvement:

    python scripts/04_survival_map.py --controller scripts/05_challenge.py

You can also run this file directly for a quick 10-second test:

    python scripts/05_challenge.py
"""
import numpy as np


# ═══════════════════════════════════════════════════════════
# YOUR CONTROLLER — Edit this section to improve performance
# ═══════════════════════════════════════════════════════════
#
# Ask Claude to try these improvements (copy-paste to Claude):
#
# Level 1 — Quick wins (expect score ~3.8-4.2):
#   "Enable velocity feedback — set kd to a small positive value like 0.5"
#   "Try different combinations of kp and kd values"
#
# Level 2 — Smarter control (expect score ~4.2-5.0):
#   "Use different gain values for the X and Y directions"
#   "Make the correction stronger when the ball is near the edge of the plate"
#
# Level 3 — Advanced (expect score ~5.0+):
#   "Add a slow integral correction to fix the ball drifting to one side"
#   "Limit the maximum correction so the controller doesn't overreact"
#
# After editing, measure your improvement:
#   python scripts/04_survival_map.py --controller scripts/05_challenge.py
# ═══════════════════════════════════════════════════════════


class PIDController:
    """Simple PID controller. You can modify this too."""

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


def make_controller(model, dt, home):
    """Create a controller function. Called once per trial.

    DO NOT change this function's name or arguments.
    Edit the logic INSIDE this function.

    Arguments (provided by the evaluator):
        model: MuJoCo model object
        dt:    simulation timestep (0.005 seconds)
        home:  list of 7 home joint positions

    Must return a function: controller(data, plate_id, ball_id, step, t)
    that sets data.ctrl[5] and data.ctrl[6] (wrist joint commands).
    """
    # --- Your gains here ---
    kp = 2.0   # How strongly to react to ball position
    kd = 0.0   # How strongly to react to ball velocity (try > 0!)

    pid_x = PIDController(kp, 0.0, kd, dt)
    pid_y = PIDController(kp, 0.0, kd, dt)

    def controller(data, plate_id, ball_id, step, t):
        brel = data.xpos[ball_id] - data.xpos[plate_id]
        ex, ey = brel[0], brel[1]

        correction_x = pid_x.compute(ex)
        correction_y = pid_y.compute(ey)

        data.ctrl[5] = home[5] + correction_x
        data.ctrl[6] = home[6] + correction_y

    return controller


# ═══════════════════════════════════════════════════════════
# Standalone runner — only executes when run directly
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    import sys
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import argparse
    import mujoco
    from mujoco_streamer import LiveStreamer

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir)

    # --- CLI ---
    parser = argparse.ArgumentParser(
        description="Quick 10s test of your controller")
    parser.add_argument("--port", type=int, default=None,
                        help="Streaming port (default: STREAM_PORT env var)")
    args = parser.parse_args()

    print("Running quick test of your controller. For the full survival map, use:")
    print("  python scripts/04_survival_map.py --controller scripts/05_challenge.py")
    print()

    # --- Load model ---
    model = mujoco.MjModel.from_xml_path(
        os.path.join(_project_root, "content", "panda_ball_balance.xml"))
    dt = model.opt.timestep

    plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    ball_qpos_addr = model.jnt_qposadr[ball_joint_id]
    ball_qvel_addr = model.jnt_dofadr[ball_joint_id]

    home = [0.0, -0.785, 0.0, -2.356, 1.184, 3.184, 1.158]
    joint_names = [f"joint{i}" for i in range(1, 8)]

    # --- Create data and set home pose ---
    data = mujoco.MjData(model)
    for jn, val in zip(joint_names, home):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        data.qpos[model.jnt_qposadr[jid]] = val
    for i, val in enumerate(home):
        data.ctrl[i] = val
    data.ctrl[7] = 0.008
    mujoco.mj_forward(model, data)

    # Place ball on plate
    data.qpos[ball_qpos_addr:ball_qpos_addr + 3] = data.xpos[plate_id] + [0, 0, 0.025]
    data.qpos[ball_qpos_addr + 3:ball_qpos_addr + 7] = [1, 0, 0, 0]
    data.qvel[ball_qvel_addr:ball_qvel_addr + 6] = 0
    mujoco.mj_forward(model, data)

    # --- Renderer and streamer ---
    renderer = mujoco.Renderer(model, height=480, width=640)
    port_kwarg = {}
    if args.port is not None:
        port_kwarg["port"] = args.port
    streamer = LiveStreamer(**port_kwarg)
    streamer.start()
    cam = streamer.make_free_camera(model)

    # --- Simulation parameters ---
    duration = 10.0
    steps = int(duration / dt)
    fps = 30
    render_every = int(1.0 / (fps * dt))

    try:
        while True:
            # Reset scene
            mujoco.mj_resetData(model, data)
            for jn, val in zip(joint_names, home):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                data.qpos[model.jnt_qposadr[jid]] = val
            for i, val in enumerate(home):
                data.ctrl[i] = val
            data.ctrl[7] = 0.008
            mujoco.mj_forward(model, data)

            data.qpos[ball_qpos_addr:ball_qpos_addr + 3] = data.xpos[plate_id] + [0, 0, 0.025]
            data.qpos[ball_qpos_addr + 3:ball_qpos_addr + 7] = [1, 0, 0, 0]
            data.qvel[ball_qvel_addr:ball_qvel_addr + 6] = 0
            mujoco.mj_forward(model, data)

            # Create controller for this trial
            controller_fn = make_controller(model, dt, home)
            survival_time = duration

            for step in range(steps):
                mujoco.mj_step(model, data)

                # Hold non-PID joints at home
                for i in [0, 1, 2, 3, 4]:
                    data.ctrl[i] = home[i]
                data.ctrl[7] = 0.008

                # Run user controller
                t = step * dt
                controller_fn(data, plate_id, ball_id, step, t)

                # NaN check
                if np.any(np.isnan(data.xpos[ball_id])):
                    survival_time = step * dt
                    break

                # Ball off plate check
                brel = data.xpos[ball_id] - data.xpos[plate_id]
                if abs(brel[0]) > 0.14 or abs(brel[1]) > 0.14 or brel[2] < -0.02:
                    survival_time = (step + 1) * dt
                    break

                # Render
                if step % render_every == 0:
                    streamer.drain_camera_commands(model, cam, renderer.scene)
                    renderer.update_scene(data, camera=cam)
                    streamer.update(renderer.render())

            print(f"Survival Time: {survival_time:.1f} seconds")
            print()

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        streamer.stop()

"""Survival Map: sweep initial ball positions and visualize PID robustness.

Places the ball at each point on a grid of offsets from plate center,
runs a 10-second headless PID trial for each, and displays a contour
plot of survival times. Green/yellow regions survive longer; dark
regions mean the ball falls off quickly.

Run with: python scripts/05_survival_map.py --kp 2 --kd 0 [--grid 20] [--no-stream]
"""
import os
import sys
import time
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)

import argparse
import mujoco
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

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
parser = argparse.ArgumentParser(description="Survival map: sweep initial ball positions")
parser.add_argument("--kp", type=float, default=2.0,
                    help="Proportional gain (default: 2.0)")
parser.add_argument("--kd", type=float, default=0.0,
                    help="Derivative gain (default: 0.0)")
parser.add_argument("--grid", type=int, default=20,
                    help="Grid resolution NxN (default: 20)")
parser.add_argument("--port", type=int, default=None,
                    help="Streaming port (default: STREAM_PORT env var or 18080)")
parser.add_argument("--no-stream", action="store_true",
                    help="Disable live streaming; save survival_map.png instead")
args = parser.parse_args()
stream_port = args.port if args.port is not None else int(os.environ.get("STREAM_PORT", 18080))

KP = args.kp
KI = 0.0
KD = args.kd

# --- Check matplotlib early ---
if not HAS_MATPLOTLIB:
    print("ERROR: Survival map requires matplotlib. Install with: pip install matplotlib")
    sys.exit(1)

# --- Setup ---
model = mujoco.MjModel.from_xml_path(os.path.join(_project_root, "content", "panda_ball_balance.xml"))
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


def run_headless_trial(x0, y0):
    """Run one headless trial with ball at (x0, y0) offset from plate center.

    Returns survival time in seconds (max 10.0).
    """
    d = mujoco.MjData(model)
    # Set home pose
    for jn, val in zip(joint_names, home):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        d.qpos[model.jnt_qposadr[jid]] = val
    for i, val in enumerate(home):
        d.ctrl[i] = val
    d.ctrl[7] = 0.008
    mujoco.mj_forward(model, d)
    # Place ball with offset
    ba = model.jnt_qposadr[ball_joint_id]
    bv = model.jnt_dofadr[ball_joint_id]
    d.qpos[ba:ba+3] = d.xpos[plate_id] + [x0, y0, 0.025]
    d.qpos[ba+3:ba+7] = [1, 0, 0, 0]
    d.qvel[bv:bv+6] = 0
    mujoco.mj_forward(model, d)
    # Run PID
    pid_x = PIDController(KP, KI, KD, dt)
    pid_y = PIDController(KP, KI, KD, dt)
    max_steps = int(10.0 / dt)
    for step in range(max_steps):
        mujoco.mj_step(model, d)
        if np.any(np.isnan(d.xpos[ball_id])):
            return step * dt
        for i in [0, 1, 2, 3, 4]:
            d.ctrl[i] = home[i]
        d.ctrl[7] = 0.008
        brel = d.xpos[ball_id] - d.xpos[plate_id]
        ex, ey = brel[0], brel[1]
        d.ctrl[5] = home[5] + pid_x.compute(ex)
        d.ctrl[6] = home[6] + pid_y.compute(ey)
        if abs(ex) > 0.14 or abs(ey) > 0.14 or brel[2] < -0.02:
            return (step + 1) * dt
    return 10.0


def run_survival_grid(grid_n):
    """Run grid_n x grid_n headless trials. Returns (xs, ys, survival_grid)."""
    xs = np.linspace(-0.12, 0.12, grid_n)
    ys = np.linspace(-0.12, 0.12, grid_n)
    grid = np.zeros((grid_n, grid_n))
    total = grid_n * grid_n
    for i, y0 in enumerate(ys):
        for j, x0 in enumerate(xs):
            grid[i, j] = run_headless_trial(x0, y0)
        done = (i + 1) * grid_n
        print(f"  Progress: {done}/{total} trials ({done*100//total}%)")
    return xs, ys, grid


def render_survival_map(xs, ys, survival_grid):
    """Render contour plot to (H, W, 3) numpy array."""
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    XX, YY = np.meshgrid(xs * 1000, ys * 1000)  # convert to mm
    levels = np.linspace(0, 10, 21)
    cf = ax.contourf(XX, YY, survival_grid, levels=levels, cmap='viridis')
    fig.colorbar(cf, label='Survival time (s)')
    ax.set_xlabel('Initial X offset (mm)')
    ax.set_ylabel('Initial Y offset (mm)')
    ax.set_title(f'Survival Map (Kp={KP}, Kd={KD})')
    ax.set_aspect('equal')
    # Plate boundary
    rect = plt.Rectangle((-140, -140), 280, 280, fill=False,
                         edgecolor='white', linewidth=1.5, linestyle='--')
    ax.add_patch(rect)
    # Mark sweet spot
    best_idx = np.unravel_index(survival_grid.argmax(), survival_grid.shape)
    ax.plot(xs[best_idx[1]] * 1000, ys[best_idx[0]] * 1000, 'w*', markersize=15)
    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb = buf[:, :, :3].copy()
    plt.close(fig)
    return rgb


# ============================================================
# Main
# ============================================================
print(f"=== Survival Map ({args.grid}x{args.grid} grid) ===")
print(f"PID: Kp={KP}, Kd={KD} (joint6+joint7, sign=+)")
print(f"Computing survival map...")

xs, ys, grid = run_survival_grid(args.grid)

best_idx = np.unravel_index(grid.argmax(), grid.shape)
print(f"Max survival: {grid.max():.1f}s at offset "
      f"({xs[best_idx[1]]*1000:.0f}mm, {ys[best_idx[0]]*1000:.0f}mm)")
print(f"Mean survival: {grid.mean():.1f}s")
print(f"Positions surviving 10s: {(grid >= 9.9).sum()}/{args.grid**2}")

map_img = render_survival_map(xs, ys, grid)

# ============================================================
# No-stream mode: save PNG and exit
# ============================================================
if args.no_stream or not HAS_STREAMER:
    if not args.no_stream and not HAS_STREAMER:
        print("WARNING: mujoco_streamer not installed, falling back to PNG output")

    from PIL import Image
    Image.fromarray(map_img).save("survival_map.png")
    print(f"Saved: survival_map.png ({args.grid}x{args.grid} grid)")

# ============================================================
# Streaming mode: display contour plot in browser
# ============================================================
else:
    print(f"Starting live stream on port {stream_port}...")
    print("Displaying survival map. Press Ctrl+C to stop.")

    streamer = LiveStreamer(port=stream_port)
    streamer.start()

    try:
        while True:
            streamer.update(map_img)
            time.sleep(1.0 / 30)
    except KeyboardInterrupt:
        print("\nStreaming stopped.")
    finally:
        streamer.stop()

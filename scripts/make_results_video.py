"""make_results_video.py — Renders TSUCHIDA's best workshop controller (99.2%
Controller Score) as a 10-second MP4 with an animated free camera, for the
workshop retrospective report deliverable.

Output: videos/tsuchida-best-result.mp4 (covered by *.mp4 in .gitignore)

Usage:
    python scripts/make_results_video.py

This script is a one-off render pipeline. It does not modify any Sprint
script and is intentionally self-contained.
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mediapy  # noqa: E402
import mujoco  # noqa: E402

# ─── Paths ─────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
MODEL_XML = REPO / "content" / "panda_ball_balance.xml"
OUTPUT_DIR = REPO / "videos"
OUTPUT_MP4 = OUTPUT_DIR / "tsuchida-best-result.mp4"

CONTROLLER_CANDIDATES = [
    Path("/tmp/workshop-logs/engineer2/05_challenge_final.py"),
    Path(
        "/home/h_fujiwara/projects/dgx-spark-playbooks/docs/workshop-archive/"
        "engineer-logs/engineer2/05_challenge_final.py"
    ),
]

# ─── Load controller ──────────────────────────────────────────────────
def load_controller():
    for path in CONTROLLER_CANDIDATES:
        if path.exists():
            spec = importlib.util.spec_from_file_location("tsuchida_ctrl", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print(f"Loaded controller from {path}")
            return mod.make_controller
    raise FileNotFoundError(
        "TSUCHIDA controller not found. Checked:\n  "
        + "\n  ".join(str(p) for p in CONTROLLER_CANDIDATES)
    )


make_controller = load_controller()

# ─── Model & data ─────────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
data = mujoco.MjData(model)
dt = model.opt.timestep

HOME = [0.0, -0.785, 0.0, -2.356, 1.184, 3.184, 1.158]

plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
ball_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
ball_qpos_addr = model.jnt_qposadr[ball_joint_id]
ball_qvel_addr = model.jnt_dofadr[ball_joint_id]


def reset_scene():
    mujoco.mj_resetData(model, data)
    for i, val in enumerate(HOME):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}")
        data.qpos[model.jnt_qposadr[jid]] = val
        data.ctrl[i] = val
    data.ctrl[7] = 0.008  # gripper clamp on plate edge
    mujoco.mj_forward(model, data)

    # Reposition ball on plate (CLAUDE.md pattern)
    data.qpos[ball_qpos_addr:ball_qpos_addr + 3] = data.xpos[plate_id] + [0, 0, 0.025]
    data.qpos[ball_qpos_addr + 3:ball_qpos_addr + 7] = [1, 0, 0, 0]
    data.qvel[ball_qvel_addr:ball_qvel_addr + 6] = 0
    mujoco.mj_forward(model, data)


reset_scene()

# ─── Camera choreography (t_sec, azimuth_deg, elevation_deg, distance_m) ──
KEYS = [
    (0.0,  45.0, -25.0, 1.5),
    (2.0,  90.0, -15.0, 0.9),
    (5.0, 135.0, -10.0, 0.7),
    (8.0, 180.0, -20.0, 1.0),
    (10.0, 225.0, -25.0, 1.2),
]


def interp_cam(t):
    t = max(KEYS[0][0], min(t, KEYS[-1][0]))
    for i in range(len(KEYS) - 1):
        t0, a0, e0, d0 = KEYS[i]
        t1, a1, e1, d1 = KEYS[i + 1]
        if t0 <= t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return (a0 + f * (a1 - a0), e0 + f * (e1 - e0), d0 + f * (d1 - d0))
    return KEYS[-1][1:]


# ─── Renderer + camera ────────────────────────────────────────────────
WIDTH, HEIGHT, FPS = 1280, 720, 30
# Bump the model's offscreen framebuffer (default 640x480) before constructing
# the Renderer so it can render at HD resolution.
model.vis.global_.offwidth = WIDTH
model.vis.global_.offheight = HEIGHT
renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE

# ─── Simulation loop ──────────────────────────────────────────────────
DURATION = 10.0
n_steps = int(DURATION / dt)
render_every = max(1, int(round(1.0 / (FPS * dt))))

controller_fn = make_controller(model, dt, HOME)
frames = []
survival_time = DURATION

for step in range(n_steps):
    mujoco.mj_step(model, data)

    # Hold joints 1-5 at home; TSUCHIDA's controller writes ctrl[5] and ctrl[6]
    for i in (0, 1, 2, 3, 4):
        data.ctrl[i] = HOME[i]
    data.ctrl[7] = 0.008

    t = step * dt
    controller_fn(data, plate_id, ball_id, step, t)

    # Safety: NaN or ball fell off
    if np.any(np.isnan(data.xpos[ball_id])):
        survival_time = step * dt
        print(f"[warn] NaN at t={survival_time:.2f}s", file=sys.stderr)
        break
    brel = data.xpos[ball_id] - data.xpos[plate_id]
    if abs(brel[0]) > 0.14 or abs(brel[1]) > 0.14 or brel[2] < -0.02:
        survival_time = (step + 1) * dt
        print(f"[warn] Ball fell at t={survival_time:.2f}s", file=sys.stderr)
        break

    if step % render_every == 0:
        az, el, dist = interp_cam(t)
        cam.azimuth = az
        cam.elevation = el
        cam.distance = dist
        cam.lookat[:] = data.xpos[plate_id]
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())

# ─── Write MP4 ────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)
mediapy.write_video(str(OUTPUT_MP4), frames, fps=FPS)

print(f"Survival: {survival_time:.2f}s / {DURATION:.1f}s")
print(f"Wrote {len(frames)} frames to {OUTPUT_MP4}")

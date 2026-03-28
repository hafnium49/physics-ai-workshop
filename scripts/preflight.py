#!/usr/bin/env python3
"""Pre-flight sanity check for Physics-AI Workshop.

Run the night before to verify that MuJoCo, models, streamer,
and PID tuning all work correctly on the host machine.

Usage:
    cd ~/projects/physics-ai-workshop
    python scripts/preflight.py
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import time
import numpy as np

# Ensure project root is on sys.path so mujoco_streamer can be imported
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HOME_POSE = [0.0, -0.785, 0.0, -2.356, 1.184, 3.184, 1.158]

results = []  # list of (passed, label, message)


def _load_model():
    """Load the pre-assembled model and return (model, data)."""
    import mujoco
    model = mujoco.MjModel.from_xml_path("content/panda_ball_balance.xml")
    data = mujoco.MjData(model)
    return model, data


def _ids(model):
    """Return (plate_id, ball_id, ball_joint_id)."""
    import mujoco
    plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    return plate_id, ball_id, ball_joint_id


def _reset_scene(model, data):
    """Reset to home pose with ball on plate."""
    import mujoco
    mujoco.mj_resetData(model, data)
    plate_id, ball_id, ball_joint_id = _ids(model)

    # Set arm joints using proper qpos addresses (free joint shifts layout)
    joint_names = [f"joint{i}" for i in range(1, 8)]
    for jn, val in zip(joint_names, HOME_POSE):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        data.qpos[model.jnt_qposadr[jid]] = val
    for i, val in enumerate(HOME_POSE):
        data.ctrl[i] = val
    data.ctrl[7] = 0.008
    mujoco.mj_forward(model, data)

    plate_pos = data.xpos[plate_id]
    ball_qpos_addr = model.jnt_qposadr[ball_joint_id]
    data.qpos[ball_qpos_addr:ball_qpos_addr + 3] = plate_pos + np.array([0, 0, 0.025])
    data.qpos[ball_qpos_addr + 3:ball_qpos_addr + 7] = [1, 0, 0, 0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    return plate_id, ball_id, ball_joint_id


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_0_mujoco_gl():
    val = os.environ.get("MUJOCO_GL")
    if val == "egl":
        return True, "MUJOCO_GL=egl"
    return False, "Set MUJOCO_GL=egl before running this script"


def check_1_model_loads():
    import mujoco
    model = mujoco.MjModel.from_xml_path("content/panda_ball_balance.xml")
    nb, nj, nu = model.nbody, model.njnt, model.nu
    ok = nb == 14 and nj == 10 and nu == 8
    msg = f"Model loads ({nb} bodies, {nj} joints, {nu} actuators)"
    if not ok:
        msg += f" — expected 14/10/8"
    return ok, msg


def check_2_ball_positioning():
    import mujoco
    model, data = _load_model()
    plate_id, ball_id, ball_joint_id = _reset_scene(model, data)
    z_gap = data.xpos[ball_id][2] - data.xpos[plate_id][2]
    ok = 0.02 < z_gap < 0.03
    msg = f"Ball positioning (z_gap={z_gap:.4f})"
    if not ok:
        msg += " — expected 0.02 < z_gap < 0.03"
    return ok, msg


def check_3_streamer_lifecycle():
    from mujoco_streamer import LiveStreamer
    s = LiveStreamer(port=0)
    s.start()
    time.sleep(0.3)
    s.stop()
    return True, "Streamer lifecycle (start/stop)"


def check_4_mediapy_import():
    import mediapy  # noqa: F401
    return True, "mediapy import"


def check_5_correct_pid():
    import mujoco
    model, data = _load_model()
    plate_id, ball_id, ball_joint_id = _reset_scene(model, data)
    dt = model.opt.timestep

    prev_ex, prev_ey = 0.0, 0.0
    survival = 10.0

    for step in range(int(10.0 / dt)):
        mujoco.mj_step(model, data)
        for i, val in enumerate(HOME_POSE):
            data.ctrl[i] = val
        data.ctrl[7] = 0.008

        brel = data.xpos[ball_id] - data.xpos[plate_id]
        ex, ey = brel[0], brel[1]
        dx = (ex - prev_ex) / dt
        dy = (ey - prev_ey) / dt

        data.ctrl[5] = HOME_POSE[5] + (2.0 * ex)   # joint6 for X
        data.ctrl[6] = HOME_POSE[6] + (2.0 * ey)   # joint7 for Y
        prev_ex, prev_ey = ex, ey

        if abs(ex) > 0.14 or abs(ey) > 0.14 or brel[2] < -0.02:
            survival = (step + 1) * dt
            break

    ok = survival >= 10.0
    msg = f"Correct PID survival ({survival:.1f}s)"
    if not ok:
        msg += f" < 10.0s required"
    return ok, msg


def check_6_wrong_pid():
    import mujoco
    model, data = _load_model()
    plate_id, ball_id, ball_joint_id = _reset_scene(model, data)
    dt = model.opt.timestep

    prev_ex, prev_ey = 0.0, 0.0
    survival = 10.0

    for step in range(int(10.0 / dt)):
        mujoco.mj_step(model, data)
        for i, val in enumerate(HOME_POSE):
            data.ctrl[i] = val
        data.ctrl[7] = 0.008

        brel = data.xpos[ball_id] - data.xpos[plate_id]
        ex, ey = brel[0], brel[1]
        dx = (ex - prev_ex) / dt
        dy = (ey - prev_ey) / dt

        data.ctrl[5] = HOME_POSE[5] - (2.0 * ex)   # wrong sign
        data.ctrl[6] = HOME_POSE[6] - (2.0 * ey)   # wrong sign
        prev_ex, prev_ey = ex, ey

        if abs(ex) > 0.14 or abs(ey) > 0.14 or brel[2] < -0.02:
            survival = (step + 1) * dt
            break

    ok = survival < 2.0
    msg = f"Wrong PID fails ({survival:.1f}s < 2.0s)"
    if not ok:
        msg += f" — expected < 2.0s"
    return ok, msg


def check_7_joint_authority():
    import mujoco
    model, data = _load_model()
    plate_id, ball_id, ball_joint_id = _ids(model)

    # Get joint qpos addresses (can't use raw index — free joint shifts qpos layout)
    joint_names = [f"joint{i}" for i in range(1, 8)]
    joint_qpos_addrs = {}
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        joint_qpos_addrs[jn] = model.jnt_qposadr[jid]

    deltas = {}
    for jnum in [5, 6, 7]:  # joint5, joint6, joint7
        jn = f"joint{jnum}"
        mujoco.mj_resetData(model, data)
        for name, val in zip(joint_names, HOME_POSE):
            data.qpos[joint_qpos_addrs[name]] = val
        mujoco.mj_forward(model, data)
        base_pos = data.xpos[plate_id].copy()

        data.qpos[joint_qpos_addrs[jn]] += 0.01
        mujoco.mj_forward(model, data)
        diff = data.xpos[plate_id] - base_pos
        dxy = np.sqrt(diff[0] ** 2 + diff[1] ** 2)
        deltas[jnum] = dxy

    ok = deltas[6] > 0.001 and deltas[7] < deltas[5]
    msg = (
        f"Joint authority "
        f"(j5={deltas[5]:.4f}, j6={deltas[6]:.4f}, j7={deltas[7]:.4f})"
    )
    if not ok:
        msg += " — expected j6>0.001, j7<j5"
    return ok, msg


def check_8_egl_rendering():
    import mujoco
    model, data = _load_model()
    _reset_scene(model, data)
    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data)
    frame = renderer.render()
    renderer.close()
    ok = frame.shape == (480, 640, 3)
    msg = f"EGL rendering ({frame.shape[0]}x{frame.shape[1]}x{frame.shape[2]})"
    if not ok:
        msg += f" — expected (480, 640, 3)"
    return ok, msg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CHECKS = [
    ("MUJOCO_GL=egl", check_0_mujoco_gl),
    ("Model loads", check_1_model_loads),
    ("Ball positioning", check_2_ball_positioning),
    ("Streamer lifecycle", check_3_streamer_lifecycle),
    ("mediapy import", check_4_mediapy_import),
    ("Correct PID", check_5_correct_pid),
    ("Wrong PID", check_6_wrong_pid),
    ("Joint authority", check_7_joint_authority),
    ("EGL rendering", check_8_egl_rendering),
]


def main():
    print("Pre-flight check: Physics-AI Workshop")
    print("=" * 38)

    failures = 0
    for label, fn in CHECKS:
        try:
            passed, msg = fn()
        except Exception as exc:
            passed = False
            msg = f"{label} — {type(exc).__name__}: {exc}"

        tag = "[PASS]" if passed else "[FAIL]"
        print(f"{tag} {msg}")
        if not passed:
            failures += 1

    print("=" * 38)
    if failures == 0:
        print("ALL CHECKS PASSED")
    else:
        print(f"{failures} CHECK{'S' if failures > 1 else ''} FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

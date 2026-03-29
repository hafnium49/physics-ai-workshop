"""ステップ2: 統合モデル panda_ball_balance.xml の検証。

モデルを読み込み、アームをホームポーズに設定し、ボールをプレートに配置して、
MJPEGライブ配信または短い.mp4として保存します。

デフォルト:  MUJOCO_GL=egl python scripts/01_validate_assembly.py        （ライブ配信）
フォールバック: MUJOCO_GL=egl python scripts/01_validate_assembly.py --no-stream --duration 3
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
# オプション: ストリーマーのインポート
# ---------------------------------------------------------------------------
try:
    from mujoco_streamer import LiveStreamer
    HAS_STREAMER = True
except ImportError:
    HAS_STREAMER = False

# ---------------------------------------------------------------------------
# コマンドライン引数
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Panda + ボール・オン・プレート組立モデルの検証")
parser.add_argument("--no-stream", action="store_true",
                    help="ライブ配信を無効化し、.mp4として保存")
parser.add_argument("--port", type=int, default=None,
                    help="MJPEG配信ポート（デフォルト: STREAM_PORT環境変数または18080）")
parser.add_argument("--duration", type=float, default=3.0,
                    help="動画の長さ（秒、デフォルト: 3.0）")
args = parser.parse_args()

stream_port = args.port if args.port is not None else int(os.environ.get("STREAM_PORT", 18080))

use_stream = (not args.no_stream) and HAS_STREAMER

# ---------------------------------------------------------------------------
# モデルの読み込み
# ---------------------------------------------------------------------------
model = mujoco.MjModel.from_xml_path(os.path.join(_project_root, "content", "panda_ball_balance.xml"))
data = mujoco.MjData(model)

# ボディID
plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")

# 関節IDとqposアドレス
joint_names = [f"joint{i}" for i in range(1, 8)]
joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
joint_addrs = [model.jnt_qposadr[jid] for jid in joint_ids]

# ホームポーズ: j5, j6, j7 はプレートの把持姿勢に合わせて調整済み
home = [0.0, -0.785, 0.0, -2.356, 1.184, 3.184, 1.158]

ball_qpos_addr = model.jnt_qposadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
]
ball_qvel_addr = model.jnt_dofadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
]


def reset_scene():
    """アームをホームポーズに設定し、ボールをプレートに配置し、速度をゼロにする。"""
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

print(f"プレート位置（世界座標）: {data.xpos[plate_id]}")
print(f"ボール位置（世界座標）:  {data.xpos[ball_id]}")
print(f"ボール・プレート間Z距離: {data.xpos[ball_id][2] - data.xpos[plate_id][2]:.4f} m")

# ---------------------------------------------------------------------------
# レンダラー（サイドカメラ）
# ---------------------------------------------------------------------------
renderer = mujoco.Renderer(model, height=480, width=640)
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side")

fps = 30
render_every = int(1.0 / (fps * model.opt.timestep))


def ball_on_plate_check():
    """ボールがおおよそプレートの上にある場合にTrueを返す。"""
    rel = data.xpos[ball_id] - data.xpos[plate_id]
    return abs(rel[0]) < 0.15 and abs(rel[1]) < 0.15 and rel[2] > -0.02


def print_diagnostics(step):
    """プレート・ボールの位置とプレート上の状態を表示する。"""
    ball_rel = data.xpos[ball_id] - data.xpos[plate_id]
    print(f"[t={data.time:.1f}s] "
          f"plate={data.xpos[plate_id]}  "
          f"ball={data.xpos[ball_id]}  "
          f"rel=({ball_rel[0]:+.4f}, {ball_rel[1]:+.4f}, {ball_rel[2]:+.4f})  "
          f"プレート上={ball_on_plate_check()}")


# ---------------------------------------------------------------------------
# シミュレーション
# ---------------------------------------------------------------------------
if use_stream:
    # ---- ライブMJPEG配信モード ----
    streamer = LiveStreamer(port=stream_port)
    streamer.start()
    cam = streamer.make_free_camera(model)
    print(f"\nライブ配信を開始しました http://localhost:{stream_port}")
    print("Ctrl+C で停止できます。\n")

    step = 0
    diag_interval = int(1.0 / model.opt.timestep)  # 約1秒ごと
    try:
        while True:  # 外側ループ: NaN発生時のリカバリ
            while True:  # 内側ループ: シミュレーションステップ
                mujoco.mj_step(model, data)

                # アームをホームに保持
                for i, val in enumerate(home):
                    data.ctrl[i] = val

                # NaNチェック — 終了せずシーンをリセット
                if np.any(np.isnan(data.xpos[ball_id])):
                    print(f"警告: ステップ {step} でNaNを検出、シーンをリセットします...")
                    break

                # フレームをレンダリングして送信
                if step % render_every == 0:
                    streamer.drain_camera_commands(model, cam, renderer.scene)
                    renderer.update_scene(data, camera=cam)
                    streamer.update(renderer.render())

                # 毎秒の診断出力
                if step % diag_interval == 0:
                    print_diagnostics(step)

                step += 1

            # NaNからリカバリ: リセットして配信を継続
            reset_scene()
            step = 0
    except KeyboardInterrupt:
        print("\n配信を停止中...")
    finally:
        streamer.stop()
        print("配信を停止しました。")

else:
    # ---- .mp4 フォールバックモード ----
    import mediapy

    if not args.no_stream and not HAS_STREAMER:
        print("警告: mujoco_streamerがインストールされていません。.mp4出力にフォールバックします。\n")

    duration = args.duration
    steps = int(duration / model.opt.timestep)
    diag_interval = int(1.0 / model.opt.timestep)  # 約1秒ごと
    frames = []

    step = 0
    while step < steps:
        mujoco.mj_step(model, data)

        # アームをホームに保持
        for i, val in enumerate(home):
            data.ctrl[i] = val

        # NaNチェック — 終了せずシーンをリセット
        if np.any(np.isnan(data.xpos[ball_id])):
            print(f"警告: ステップ {step} でNaNを検出、シーンをリセットします...")
            reset_scene()
            step = 0
            frames.clear()
            continue

        # フレームのレンダリング
        if step % render_every == 0:
            renderer.update_scene(data, camera=cam_id)
            frames.append(renderer.render())

        # 毎秒の診断出力
        if step % diag_interval == 0:
            print_diagnostics(step)

        step += 1

    # 最終診断
    ball_rel = data.xpos[ball_id] - data.xpos[plate_id]
    print(f"\n最終プレート位置: {data.xpos[plate_id]}")
    print(f"最終ボール位置:  {data.xpos[ball_id]}")
    print(f"ボールのプレートに対する相対位置: x={ball_rel[0]:.4f} y={ball_rel[1]:.4f} z={ball_rel[2]:.4f}")
    print(f"プレート上: {ball_on_plate_check()}")

    mediapy.write_video("assembly_test.mp4", frames, fps=fps)
    print(f"\n動画を保存しました: assembly_test.mp4 ({len(frames)} フレーム)")

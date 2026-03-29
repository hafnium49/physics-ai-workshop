"""ステップ4: 10秒間の解が存在することを検証する。

関節ペアとPID符号・ゲインの組み合わせを体系的にテストする。
タスクが解決可能であることを証明し、正しい制御アーキテクチャを特定する。

デフォルト:  python scripts/03_optimize_pid.py               （グリッド探索 + 最良結果を配信）
フォールバック: python scripts/03_optimize_pid.py --no-stream   （グリッド探索 + 最良結果を.mp4で保存）
ドライラン:  python scripts/03_optimize_pid.py --no-render   （グリッド探索のみ、映像出力なし）
"""
import os
import sys
os.environ.setdefault("MUJOCO_GL", "egl")  # ヘッドレス描画用（GPUでオフスクリーンレンダリング）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # プロジェクトルートをインポートパスに追加

import argparse
import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# オプション: ストリーマーのインポート
# ---------------------------------------------------------------------------
try:
    from mujoco_streamer import LiveStreamer  # ブラウザへのリアルタイム映像配信ライブラリ
    HAS_STREAMER = True
except ImportError:
    HAS_STREAMER = False  # ライブ配信が使えない場合は.mp4保存にフォールバック

# ---------------------------------------------------------------------------
# コマンドライン引数
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="ボール・オン・プレートのPID最適化グリッド探索")
parser.add_argument("--no-stream", action="store_true",
                    help="ライブ配信を無効化し、.mp4として保存")
parser.add_argument("--no-render", action="store_true",
                    help="全レンダリングをスキップ（ドライラン）")
parser.add_argument("--port", type=int, default=None,
                    help="MJPEG配信ポート（デフォルト: STREAM_PORT環境変数または18080）")
args = parser.parse_args()
stream_port = args.port if args.port is not None else int(os.environ.get("STREAM_PORT", 18080))  # 各参加者固有のポート番号を取得

# ---------------------------------------------------------------------------
# モデルの読み込み
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)

model = mujoco.MjModel.from_xml_path(os.path.join(_project_root, "content", "panda_ball_balance.xml"))

plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
ball_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")

home = [0.0, -0.785, 0.0, -2.356, 1.184, 3.184, 1.158]
joint_names = [f"joint{i}" for i in range(1, 8)]


def run_trial(joint_x_idx, joint_y_idx, sign, kp, kd, duration=10.0, render=False):
    """指定された関節ペア、符号、PIDゲインでシミュレーション試行を実行する。

    引数:
        joint_x_idx: X軸制御のアクチュエータインデックス（0始まり）
        joint_y_idx: Y軸制御のアクチュエータインデックス（0始まり）
        sign: 補正方向の符号（+1 または -1）
        kp, kd: PIDゲイン
        duration: シミュレーション時間（秒）
        render: Trueの場合、(survival_time, frames)を返す

    戻り値:
        survival_time (float)、render=Trueの場合は (survival_time, frames)
    """
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    # アームをホームポーズに設定
    for jn, val in zip(joint_names, home):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        data.qpos[model.jnt_qposadr[jid]] = val
    for i, val in enumerate(home):
        data.ctrl[i] = val
    data.ctrl[7] = 0.008
    mujoco.mj_forward(model, data)

    # ボールをプレートに配置
    ba = model.jnt_qposadr[ball_joint_id]
    bv = model.jnt_dofadr[ball_joint_id]
    data.qpos[ba:ba + 3] = data.xpos[plate_id] + [0, 0, 0.025]
    data.qpos[ba + 3:ba + 7] = [1, 0, 0, 0]
    data.qvel[bv:bv + 6] = 0
    mujoco.mj_forward(model, data)

    renderer = None
    frames = []
    cam_id = -1
    if render:
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side")

    fps = 30
    render_every = int(1.0 / (fps * dt))
    prev_ex, prev_ey = 0.0, 0.0
    steps = int(duration / dt)

    for step in range(steps):
        mujoco.mj_step(model, data)

        # 全関節をホームに保持
        for i, val in enumerate(home):
            data.ctrl[i] = val
        data.ctrl[7] = 0.008

        # ボール誤差
        brel = data.xpos[ball_id] - data.xpos[plate_id]
        ex, ey = brel[0], brel[1]
        dx = (ex - prev_ex) / dt
        dy = (ey - prev_ey) / dt

        # 選択された関節にPID補正を適用
        data.ctrl[joint_y_idx] = home[joint_y_idx] + sign * (kp * ey + kd * dy)
        data.ctrl[joint_x_idx] = home[joint_x_idx] + sign * (kp * ex + kd * dx)

        prev_ex, prev_ey = ex, ey

        # NaNチェック
        if np.any(np.isnan(data.xpos[ball_id])):
            t = step * dt
            return (t, frames) if render else t

        # ボールがプレートから落下
        if abs(ex) > 0.14 or abs(ey) > 0.14 or brel[2] < -0.02:
            t = (step + 1) * dt
            return (t, frames) if render else t

        # レンダリング
        if render and step % render_every == 0:
            renderer.update_scene(data, camera=cam_id)
            frames.append(renderer.render())

    return (duration, frames) if render else duration


try:
    # --- フェーズ1: 正しい関節ペアを見つける ---
    print("=" * 60)
    print("フェーズ1: 関節ペアの検証 (Kp=2, Kd=0)")
    print("=" * 60)

    pairings = [
        ("j6(X)+j7(Y)", 5, 6),   # 正しいペア
        ("j6(X)+j5(Y)", 5, 4),   # 代替ペア（動作する）
        ("j5(X)+j6(Y)", 4, 5),   # 軸が逆
        ("j5(X)+j4(Y)", 4, 3),   # 完全に間違った関節
    ]

    for name, jx, jy in pairings:
        for sign in [+1, -1]:
            t = run_trial(jx, jy, sign, kp=2, kd=0, duration=5.0)
            marker = " <-- 成功" if t >= 5.0 else ""
            print(f"  {name} sign={sign:+d} -> {t:.1f}s{marker}")

    # --- フェーズ2: 最良ペアでゲインを確認 ---
    print()
    print("=" * 60)
    print("フェーズ2: ゲイン探索 j6(X)+j7(Y), sign=+1")
    print("=" * 60)

    results = []
    for kp in [1, 2, 3, 5, 10]:
        for kd in [0, 1, 2, 5]:
            t = run_trial(5, 6, +1, kp, kd)
            results.append((kp, kd, t))
            marker = " ***" if t >= 10.0 else ""
            print(f"  Kp={kp:>3d} Kd={kd:>2d} -> 維持時間: {t:.1f}秒{marker}")

    best = max(results, key=lambda x: x[2])
    print(f"\n最良: Kp={best[0]}, Kd={best[1]}, 維持時間={best[2]:.1f}秒")
except KeyboardInterrupt:
    print("\nグリッド探索が中断されました。")
    raise SystemExit(0)

# --- フェーズ3: 最良結果をレンダリング（--no-renderでなければ） ---
if args.no_render:
    print("\n--no-render が指定されたため、映像出力をスキップします。")
else:
    use_stream = (not args.no_stream) and HAS_STREAMER

    if use_stream:
        # ---- 最良結果のライブMJPEG配信 ----
        print(f"\nポート {stream_port} で最良結果を配信中...")
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side")
        streamer = LiveStreamer(port=stream_port)
        streamer.start()
        cam = streamer.make_free_camera(model)

        data = mujoco.MjData(model)
        dt = model.opt.timestep
        ba = model.jnt_qposadr[ball_joint_id]
        bv = model.jnt_dofadr[ball_joint_id]
        fps = 30
        render_every = int(1.0 / (fps * dt))
        steps = int(10.0 / dt)

        print(f"配信中 Kp={best[0]}, Kd={best[1]}（自動リセットループ）")
        print("Ctrl+C で停止できます。\n")

        try:
            while True:
                # 各イテレーションでシーンをリセット
                mujoco.mj_resetData(model, data)
                for jn, val in zip(joint_names, home):
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                    data.qpos[model.jnt_qposadr[jid]] = val
                for i, val in enumerate(home):
                    data.ctrl[i] = val
                data.ctrl[7] = 0.008
                mujoco.mj_forward(model, data)

                data.qpos[ba:ba + 3] = data.xpos[plate_id] + [0, 0, 0.025]
                data.qpos[ba + 3:ba + 7] = [1, 0, 0, 0]
                data.qvel[bv:bv + 6] = 0
                mujoco.mj_forward(model, data)
                prev_ex, prev_ey = 0.0, 0.0

                for step in range(steps):
                    mujoco.mj_step(model, data)

                    for i, val in enumerate(home):
                        data.ctrl[i] = val
                    data.ctrl[7] = 0.008

                    brel = data.xpos[ball_id] - data.xpos[plate_id]
                    ex, ey = brel[0], brel[1]
                    dx = (ex - prev_ex) / dt
                    dy = (ey - prev_ey) / dt

                    data.ctrl[5] = home[5] + (best[0] * ex + best[1] * dx)
                    data.ctrl[6] = home[6] + (best[0] * ey + best[1] * dy)

                    prev_ex, prev_ey = ex, ey

                    if np.any(np.isnan(data.xpos[ball_id])):
                        break
                    if abs(ex) > 0.14 or abs(ey) > 0.14 or brel[2] < -0.02:
                        break

                    if step % render_every == 0:
                        streamer.drain_camera_commands(model, cam, renderer.scene)
                        renderer.update_scene(data, camera=cam)
                        streamer.update(renderer.render())
        except KeyboardInterrupt:
            print("\n配信を停止しました。")
        finally:
            streamer.stop()

    else:
        # ---- .mp4 フォールバックモード ----
        import mediapy

        if not args.no_stream and not HAS_STREAMER:
            print("警告: mujoco_streamerがインストールされていません。.mp4出力にフォールバックします")

        print("\n最良結果をレンダリング中...")
        t, frames = run_trial(5, 6, +1, best[0], best[1], render=True)
        mediapy.write_video("best_balance.mp4", frames, fps=30)
        print(f"動画を保存しました: best_balance.mp4 ({len(frames)} フレーム, 維持時間 {t:.1f}秒)")

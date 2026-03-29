"""ステップ3: ボール・オン・プレートのベースラインPIDコントローラー。

joint6/joint7にPIDコントローラーを適用し、ボールをプレート中心に維持します。
維持時間をターミナルに表示します。意図的に符号を反転させており、ボールはすぐに落下します。

デフォルト: ボール落下時に自動リセットするライブMJPEG配信。
フォールバック: --no-stream で10秒間の試行を.mp4として保存。

実行方法: python scripts/02_pid_baseline.py [--no-stream] [--port 18080] [--kp 50] [--kd 10]
"""
import os
import sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)

import argparse
import mujoco
import numpy as np

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


# --- コマンドライン引数 ---
parser = argparse.ArgumentParser(description="ボール・オン・プレートのベースラインPIDコントローラー")
parser.add_argument("--no-stream", action="store_true",
                    help="ライブ配信を無効化し、.mp4として保存")
parser.add_argument("--port", type=int, default=None,
                    help="MJPEG配信ポート（デフォルト: STREAM_PORT環境変数または18080）")
parser.add_argument("--kp", type=float, default=50.0,
                    help="比例ゲイン（デフォルト: 50.0）")
parser.add_argument("--kd", type=float, default=10.0,
                    help="微分ゲイン（デフォルト: 10.0）")
args = parser.parse_args()
stream_port = args.port if args.port is not None else int(os.environ.get("STREAM_PORT", 18080))

KP = args.kp
KI = 0.0
KD = args.kd

# --- セットアップ ---
model = mujoco.MjModel.from_xml_path(os.path.join(_project_root, "content", "panda_ball_balance.xml"))
data = mujoco.MjData(model)
dt = model.opt.timestep

plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")

# ホームポーズ（j5, j6, j7 は新しいプレート姿勢に合わせて調整済み）
home = [0.0, -0.785, 0.0, -2.356, 1.184, 3.184, 1.158]
joint_names = [f"joint{i}" for i in range(1, 8)]

# 関節IDとアドレス（診断用にキャッシュ）
joint_ids = {}
for jn in joint_names:
    joint_ids[jn] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)

ball_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
ball_qpos_addr = model.jnt_qposadr[ball_joint_id]
ball_qvel_addr = model.jnt_dofadr[ball_joint_id]

# レンダラー
renderer = mujoco.Renderer(model, height=480, width=640)
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side")


def reset_scene(m, d):
    """アームをホームポーズにリセットし、ボールをプレートに配置する。"""
    mujoco.mj_resetData(m, d)
    for jn, val in zip(joint_names, home):
        jid = joint_ids[jn]
        d.qpos[m.jnt_qposadr[jid]] = val
    for i, val in enumerate(home):
        d.ctrl[i] = val
    d.ctrl[7] = 0.008  # グリッパーを閉じる
    mujoco.mj_forward(m, d)

    # ボールをプレートに配置
    d.qpos[ball_qpos_addr:ball_qpos_addr + 3] = d.xpos[plate_id] + [0, 0, 0.025]
    d.qpos[ball_qpos_addr + 3:ball_qpos_addr + 7] = [1, 0, 0, 0]
    d.qvel[ball_qvel_addr:ball_qvel_addr + 6] = 0
    mujoco.mj_forward(m, d)


def run_joint_diagnostics(m, d):
    """手首関節5, 6, 7の関節権限診断。

    各手首関節を+0.01 rad微動させ、プレート位置の変化を測定する。
    どの関節が実際にプレートを傾けるかを特定するのに役立つ。
    正解: joint6 (ctrl[5]) がX軸、joint7 (ctrl[6]) がY軸を傾ける。
    """
    # 現在の状態を保存
    qpos_save = d.qpos.copy()
    qvel_save = d.qvel.copy()
    ctrl_save = d.ctrl.copy()

    # 基準プレート位置を取得
    mujoco.mj_forward(m, d)
    plate_pos_base = d.xpos[plate_id].copy()

    parts = []
    for jnum in [5, 6, 7]:
        jname = f"joint{jnum}"
        jid = joint_ids[jname]
        qadr = m.jnt_qposadr[jid]

        # 微動
        d.qpos[:] = qpos_save
        d.qvel[:] = qvel_save
        d.ctrl[:] = ctrl_save
        d.qpos[qadr] += 0.01
        mujoco.mj_forward(m, d)

        plate_pos_nudged = d.xpos[plate_id].copy()
        dx = plate_pos_nudged[0] - plate_pos_base[0]
        dy = plate_pos_nudged[1] - plate_pos_base[1]
        dxy = np.sqrt(dx**2 + dy**2)

        parts.append(f"joint{jnum} +0.01 rad → plate dX={dx:+.4f} dY={dy:+.4f} dXY={dxy:.4f}")

    # 状態を復元
    d.qpos[:] = qpos_save
    d.qvel[:] = qvel_save
    d.ctrl[:] = ctrl_save
    mujoco.mj_forward(m, d)

    print(f"[診断] {parts[0]}  |  {parts[1]}  |  {parts[2]}")


def ball_off_plate(d):
    """ボールがプレートから落ちたかを確認する。"""
    ball_rel_world = d.xpos[ball_id] - d.xpos[plate_id]
    error_x = ball_rel_world[0]
    error_y = ball_rel_world[1]
    return (abs(error_x) > 0.14 or abs(error_y) > 0.14 or ball_rel_world[2] < -0.02)


def run_simulation_step(d, pid_x, pid_y):
    """PID制御で1シミュレーションステップを実行する。(error_x, error_y, nan_detected)を返す。"""
    mujoco.mj_step(model, d)

    # PID対象外の関節をホームに保持
    for i in [0, 1, 2, 3, 4]:  # joint1-5
        d.ctrl[i] = home[i]
    d.ctrl[7] = 0.008  # グリッパー

    # センシング: 世界座標系でのボールのプレートに対する相対位置
    ball_rel_world = d.xpos[ball_id] - d.xpos[plate_id]
    error_x = ball_rel_world[0]
    error_y = ball_rel_world[1]

    # NaNガード
    if np.any(np.isnan(d.xpos[ball_id])):
        return error_x, error_y, True

    correction_x = pid_x.compute(error_x)
    correction_y = pid_y.compute(error_y)

    # joint6 (ctrl[5]) と joint7 (ctrl[6]) に補正を適用
    # ===== 意図的なベースラインバグ =====
    # 正しい関節を使っているが、符号が反転（マイナス）している。
    # - マイナス符号がプレートを逆方向に押すため、
    #   ボールはほぼ即座に転がり落ちる。
    # ワークショップの課題はClaudeに以下を発見させること:
    #   1. ctrl[5]=joint6 がX軸、ctrl[6]=joint7 がY軸（これは正しい）
    #   2. 補正の符号はマイナスではなくプラスであるべき
    #   3. 符号を修正すれば Kp~2, Kd~0 で十分
    # 正しい符号と適度なゲインで、ボールは10秒維持できる。
    # ===================================
    d.ctrl[5] = home[5] - correction_x  # joint6: X軸（符号が逆！）
    d.ctrl[6] = home[6] - correction_y  # joint7: Y軸（符号が逆！）

    return error_x, error_y, False


# ============================================================
# .mp4モード: 10秒間の1回実行、動画保存して終了
# ============================================================
if args.no_stream or not HAS_STREAMER:
    import mediapy

    if not args.no_stream and not HAS_STREAMER:
        print("警告: mujoco_streamerがインストールされていません。.mp4出力にフォールバックします")

    reset_scene(model, data)
    run_joint_diagnostics(model, data)

    pid_x = PIDController(KP, KI, KD, dt)
    pid_y = PIDController(KP, KI, KD, dt)

    duration = 10.0
    fps = 30
    steps = int(duration / dt)
    render_every = int(1.0 / (fps * dt))
    frames = []
    survival_time = duration

    print(f"PIDゲイン: Kp={KP}, Ki={KI}, Kd={KD}")
    print(f"シミュレーション中 {duration}秒...")
    print()

    for step in range(steps):
        error_x, error_y, nan_detected = run_simulation_step(data, pid_x, pid_y)

        if nan_detected:
            print(f"エラー: ステップ {step} でNaN")
            survival_time = step * dt
            break

        if ball_off_plate(data):
            survival_time = (step + 1) * dt
            break

        # 定期的な診断出力
        t = (step + 1) * dt
        if step % 200 == 0:
            print(f"  t={t:.1f}s  誤差: x={error_x:+.4f} y={error_y:+.4f}  "
                  f"ctrl6={data.ctrl[5]:.3f} ctrl7={data.ctrl[6]:.3f}")

        # レンダリング
        if step % render_every == 0:
            renderer.update_scene(data, camera=cam_id)
            frames.append(renderer.render())

    print()
    print(f"維持時間: {survival_time:.1f} 秒")

    mediapy.write_video("attempt_1.mp4", frames, fps=fps)
    print(f"動画を保存しました: attempt_1.mp4 ({len(frames)} フレーム)")

# ============================================================
# 配信モード: 自動リセットループ付きライブMJPEG
# ============================================================
else:
    print(f"ポート {stream_port} でライブ配信を開始...")
    print(f"PIDゲイン: Kp={KP}, Ki={KI}, Kd={KD}")
    print(f"Ctrl+C で停止できます。\n")

    streamer = LiveStreamer(port=stream_port)
    streamer.start()
    cam = streamer.make_free_camera(model)
    fps = 30
    render_every = int(1.0 / (fps * dt))
    attempt = 0

    try:
        while True:
            # 新しい試行のためにリセット
            attempt += 1
            reset_scene(model, data)
            run_joint_diagnostics(model, data)

            pid_x = PIDController(KP, KI, KD, dt)
            pid_y = PIDController(KP, KI, KD, dt)

            print(f"--- 試行 {attempt} ---")
            step = 0
            survival_time = 0.0

            while True:
                error_x, error_y, nan_detected = run_simulation_step(data, pid_x, pid_y)
                step += 1

                if nan_detected:
                    print(f"エラー: ステップ {step} でNaN")
                    survival_time = step * dt
                    break

                if ball_off_plate(data):
                    survival_time = (step) * dt
                    break

                # 定期的な診断出力
                t = step * dt
                if step % 200 == 0:
                    print(f"  t={t:.1f}s  誤差: x={error_x:+.4f} y={error_y:+.4f}  "
                          f"ctrl6={data.ctrl[5]:.3f} ctrl7={data.ctrl[6]:.3f}")

                # フレームを配信
                if step % render_every == 0:
                    streamer.drain_camera_commands(model, cam, renderer.scene)
                    renderer.update_scene(data, camera=cam)
                    streamer.update(renderer.render())

            print(f"維持時間: {survival_time:.1f} 秒")

            # リセット前にボールが床に落ちるのを待つ
            if not nan_detected:
                max_fall = int(3.0 / dt)
                settle = int(0.5 / dt)
                landed_at = None
                for fall_step in range(max_fall):
                    mujoco.mj_step(model, data)
                    for i in range(7):
                        data.ctrl[i] = home[i]
                    data.ctrl[7] = 0.008
                    if fall_step % render_every == 0:
                        streamer.drain_camera_commands(model, cam, renderer.scene)
                        renderer.update_scene(data, camera=cam)
                        streamer.update(renderer.render())
                    ball_z = data.xpos[ball_id][2]
                    if landed_at is None and ball_z < 0.05:
                        landed_at = fall_step
                    if landed_at is not None and (fall_step - landed_at) >= settle:
                        break
                    if np.any(np.isnan(data.xpos[ball_id])):
                        break
            print()

    except KeyboardInterrupt:
        print("\n配信を停止しました。")
    finally:
        streamer.stop()

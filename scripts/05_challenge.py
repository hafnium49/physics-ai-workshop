"""コントローラー改善プレイグラウンド — ボールオンプレートのコントローラーを改善しよう。

このファイルがあなたのコントローラーです。make_controller()関数を編集して
様々な制御戦略を試し、改善度を測定しましょう:

    python scripts/04_survival_map.py --controller scripts/05_challenge.py

このファイルを直接実行して10秒間の簡易テストもできます:

    python scripts/05_challenge.py
"""
import numpy as np


# ═══════════════════════════════════════════════════════════
# コントローラー — このセクションを編集して改善しよう
# ═══════════════════════════════════════════════════════════
#
# 以下の改善案をClaudeにコピー＆ペーストしてください:
#
# レベル1 — すぐできる改善（スコア ~3.8-4.2）:
#   「kdを0.5くらいにして速度フィードバックを有効にして」
#   「kpとkdの組み合わせをいろいろ試して」
#
# レベル2 — もう少し賢い制御（スコア ~4.2-5.0）:
#   「X方向とY方向で異なるゲインを使って」
#   「ボールがプレートの端に近いときは補正を強くして」
#
# レベル3 — 上級（スコア ~5.0以上）:
#   「ボールが片側にずれていくのを積分補正で直して」
#   「過剰な補正を防ぐためにリミッターをつけて」
#
# 編集後、改善度を測定:
#   python scripts/04_survival_map.py --controller scripts/05_challenge.py
# ═══════════════════════════════════════════════════════════


class PIDController:
    """シンプルなPIDコントローラー。これも変更可能です。"""

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
    """コントローラー関数を作成。試行ごとに1回呼び出されます。

    この関数の名前や引数は変更しないでください。
    この関数の中身のロジックを編集してください。

    引数（評価器から提供されます）:
        model: MuJoCoモデルオブジェクト
        dt:    シミュレーションのタイムステップ（0.005秒）
        home:  7つのホーム関節位置のリスト

    戻り値: controller(data, plate_id, ball_id, step, t) 関数
    data.ctrl[5]とdata.ctrl[6]（手首関節のコマンド）を設定する。
    """
    # --- ゲインをここで設定 ---
    kp = 2.0   # ボール位置への反応の強さ
    kd = 0.0   # ボール速度への反応の強さ（0より大きくしてみよう！）

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
# 以下は編集しないでください — テスト用スタンドアロンランナー
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
        description="コントローラーの10秒間簡易テスト")
    parser.add_argument("--port", type=int, default=None,
                        help="配信ポート（デフォルト: STREAM_PORT環境変数）")
    args = parser.parse_args()

    print("コントローラーの簡易テストを実行中。完全な維持マップを取得するには:")
    print("  python scripts/04_survival_map.py --controller scripts/05_challenge.py")
    print()

    # --- モデルの読み込み ---
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

    # --- データ作成とホームポーズの設定 ---
    data = mujoco.MjData(model)
    for jn, val in zip(joint_names, home):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        data.qpos[model.jnt_qposadr[jid]] = val
    for i, val in enumerate(home):
        data.ctrl[i] = val
    data.ctrl[7] = 0.008
    mujoco.mj_forward(model, data)

    # ボールをプレート上に配置
    data.qpos[ball_qpos_addr:ball_qpos_addr + 3] = data.xpos[plate_id] + [0, 0, 0.025]
    data.qpos[ball_qpos_addr + 3:ball_qpos_addr + 7] = [1, 0, 0, 0]
    data.qvel[ball_qvel_addr:ball_qvel_addr + 6] = 0
    mujoco.mj_forward(model, data)

    # --- レンダラーとストリーマー ---
    renderer = mujoco.Renderer(model, height=480, width=640)
    port_kwarg = {}
    if args.port is not None:
        port_kwarg["port"] = args.port
    streamer = LiveStreamer(**port_kwarg)
    streamer.start()
    cam = streamer.make_free_camera(model)

    # --- シミュレーションパラメータ ---
    duration = 10.0
    steps = int(duration / dt)
    fps = 30
    render_every = int(1.0 / (fps * dt))

    try:
        while True:
            # シーンをリセット
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

            # この試行用のコントローラーを作成
            controller_fn = make_controller(model, dt, home)
            survival_time = duration

            for step in range(steps):
                mujoco.mj_step(model, data)

                # PID以外の関節をホームに保持
                for i in [0, 1, 2, 3, 4]:
                    data.ctrl[i] = home[i]
                data.ctrl[7] = 0.008

                # ユーザーコントローラーを実行
                t = step * dt
                controller_fn(data, plate_id, ball_id, step, t)

                # NaNチェック
                if np.any(np.isnan(data.xpos[ball_id])):
                    survival_time = step * dt
                    break

                # ボール落下チェック
                brel = data.xpos[ball_id] - data.xpos[plate_id]
                if abs(brel[0]) > 0.14 or abs(brel[1]) > 0.14 or brel[2] < -0.02:
                    survival_time = (step + 1) * dt
                    break

                # レンダリング
                if step % render_every == 0:
                    streamer.drain_camera_commands(model, cam, renderer.scene)
                    renderer.update_scene(data, camera=cam)
                    streamer.update(renderer.render())

            print(f"維持時間: {survival_time:.1f} 秒")
            print()

    except KeyboardInterrupt:
        print("\n停止しました。")
    finally:
        streamer.stop()

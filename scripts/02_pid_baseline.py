"""ステップ3: ボール・オン・プレートのベースラインPIDコントローラー。

joint6/joint7にPIDコントローラーを適用し、ボールをプレート中心に維持します。
維持時間をターミナルに表示します。意図的に符号を反転させており、ボールはすぐに落下します。

デフォルト: ボール落下時に自動リセットするライブMJPEG配信。
フォールバック: --no-stream で10秒間の試行を.mp4として保存。

実行方法: python scripts/02_pid_baseline.py [--no-stream] [--port 18080] [--kp 50] [--kd 10]
"""
import os
import sys
# ヘッドレス環境でOpenGLレンダリングを有効にする設定（GPUオフスクリーン描画）
os.environ.setdefault("MUJOCO_GL", "egl")
# プロジェクトルートをPythonの検索パスに追加（mujoco_streamerを見つけるため）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# スクリプトの場所からプロジェクトルートを計算（モデルファイルの相対パス解決用）
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)

import argparse
import mujoco
import numpy as np

try:
    # ブラウザでリアルタイム映像を見るためのライブ配信モジュール
    from mujoco_streamer import LiveStreamer
    HAS_STREAMER = True
except ImportError:
    # インストールされていなければmp4保存にフォールバック
    HAS_STREAMER = False


# PIDコントローラー: 誤差をゼロに近づけるための制御アルゴリズム
# P（比例）= 今の誤差に比例した補正、I（積分）= 過去の誤差の累積を補正、D（微分）= 誤差の変化速度を補正
class PIDController:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp    # Kp: 比例ゲイン（誤差が大きいほど強く補正）
        self.ki = ki    # Ki: 積分ゲイン（長時間残る小さな誤差を補正）
        self.kd = kd    # Kd: 微分ゲイン（急な変化を抑制、ダンパーのような役割）
        self.dt = dt    # 制御周期（秒）— シミュレーションのtimestepと同じ
        self.integral = 0.0   # 誤差の積分値（累積）
        self.prev_error = 0.0  # 前回の誤差（微分計算用）

    def compute(self, error):
        """誤差から制御出力を計算する。error > 0 なら正方向にずれている。"""
        # I項: 誤差を時間で積分（誤差 × 制御周期 を累積加算）
        self.integral += error * self.dt
        # D項: 誤差の時間微分（今回の誤差 − 前回の誤差）÷ 制御周期
        derivative = (error - self.prev_error) / self.dt
        # PID出力 = P項 + I項 + D項
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error  # 次回のD項計算のために今回の誤差を保存
        return output

    def reset(self):
        """積分値と前回誤差をクリア（新しい試行の開始時に呼ぶ）"""
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
# 配信ポート: コマンドライン引数 → 環境変数 → デフォルト18080 の優先順位で決定
stream_port = args.port if args.port is not None else int(os.environ.get("STREAM_PORT", 18080))

KP = args.kp   # 比例ゲイン（コマンドラインから変更可能）
KI = 0.0       # 積分ゲイン（このベースラインではゼロ — 定常偏差の補正なし）
KD = args.kd   # 微分ゲイン（コマンドラインから変更可能）

# --- セットアップ ---
# XMLファイルからMuJoCoモデルを構築（ロボット・プレート・ボールの物理定義すべてを含む）
model = mujoco.MjModel.from_xml_path(os.path.join(_project_root, "content", "panda_ball_balance.xml"))
# シミュレーション状態（関節角度・速度・接触力など）を保持するデータ構造を作成
data = mujoco.MjData(model)
dt = model.opt.timestep  # シミュレーションの時間刻み（0.005秒 = 200Hz）

# 名前からID番号を取得 — 位置追跡に必要
plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")  # プレートのボディID
ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")    # ボールのボディID

# ホームポーズ（プレートが水平になる関節角度）— j5, j6, j7 は新しいプレート姿勢に合わせて調整済み
home = [0.0, -0.785, 0.0, -2.356, 1.184, 3.184, 1.158]
joint_names = [f"joint{i}" for i in range(1, 8)]  # joint1〜joint7の名前リスト

# 関節IDとアドレス（診断用にキャッシュ）— 各関節の名前→ID変換を辞書に保存
joint_ids = {}
for jn in joint_names:
    joint_ids[jn] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)  # 名前からID番号を取得

# ボールの自由関節のアドレスを取得（位置と速度の両方）
ball_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")  # 名前からID番号を取得
ball_qpos_addr = model.jnt_qposadr[ball_joint_id]  # qpos配列内のインデックス（位置x,y,z + quaternion）
ball_qvel_addr = model.jnt_dofadr[ball_joint_id]    # qvel配列内のインデックス（並進3 + 回転3 = 6自由度）

# レンダラー — 画像描画用（480×640ピクセル）
renderer = mujoco.Renderer(model, height=480, width=640)
# モデル内で定義された「side」カメラのIDを取得
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side")


def reset_scene(m, d):
    """アームをホームポーズにリセットし、ボールをプレートに配置する。"""
    # すべての状態（位置・速度・力など）を初期値に戻す
    mujoco.mj_resetData(m, d)
    # 各関節にホームポーズの角度を設定
    for jn, val in zip(joint_names, home):
        jid = joint_ids[jn]
        d.qpos[m.jnt_qposadr[jid]] = val  # 全関節の角度（ラジアン）に目標値を書き込む
    # アクチュエータの制御信号もホームポーズに合わせる（位置制御なので角度を指定）
    for i, val in enumerate(home):
        d.ctrl[i] = val  # アクチュエータi番目の制御信号
    d.ctrl[7] = 0.008  # グリッパーの指位置（プレート端を挟む）
    # 関節位置から全ボディの世界座標を計算（順運動学）— プレート位置を確定させる
    mujoco.mj_forward(m, d)

    # ボールをプレートに配置
    # ボールの位置 = プレート中心 + Z方向に0.025m（ボール半径0.02m + マージン）
    d.qpos[ball_qpos_addr:ball_qpos_addr + 3] = d.xpos[plate_id] + [0, 0, 0.025]
    # ボールの姿勢を単位クォータニオン（回転なし）に設定
    d.qpos[ball_qpos_addr + 3:ball_qpos_addr + 7] = [1, 0, 0, 0]
    # ボールの速度（並進3+回転3）をすべてゼロにする
    d.qvel[ball_qvel_addr:ball_qvel_addr + 6] = 0
    # 再度順運動学を計算して、ボール配置後の正確な世界座標を更新
    mujoco.mj_forward(m, d)


def run_joint_diagnostics(m, d):
    """手首関節5, 6, 7の関節権限診断。

    各手首関節を+0.01 rad微動させ、プレート位置の変化を測定する。
    どの関節が実際にプレートを傾けるかを特定するのに役立つ。
    正解: joint6 (ctrl[5]) がX軸、joint7 (ctrl[6]) がY軸を傾ける。
    """
    # 現在の状態を保存（診断後に元に戻すため）
    qpos_save = d.qpos.copy()  # 全関節の角度（ラジアン）を退避
    qvel_save = d.qvel.copy()  # 全関節の角速度を退避
    ctrl_save = d.ctrl.copy()  # 全アクチュエータの制御信号を退避

    # 基準プレート位置を取得 — 微動前の位置を記録
    mujoco.mj_forward(m, d)  # 関節位置から全ボディの世界座標を計算（順運動学）
    plate_pos_base = d.xpos[plate_id].copy()  # ボディの世界座標位置（3次元ベクトル）

    parts = []
    for jnum in [5, 6, 7]:  # 手首に近い3関節をテスト
        jname = f"joint{jnum}"
        jid = joint_ids[jname]
        qadr = m.jnt_qposadr[jid]  # この関節のqpos配列内のインデックス

        # 状態を元に戻してから微動させる（各テストが独立になるように）
        d.qpos[:] = qpos_save
        d.qvel[:] = qvel_save
        d.ctrl[:] = ctrl_save
        d.qpos[qadr] += 0.01  # 0.01ラジアン（約0.57度）だけ関節を動かす
        # 関節位置から全ボディの世界座標を計算（順運動学）
        mujoco.mj_forward(m, d)

        # 微動後のプレート位置と基準位置の差を計算
        plate_pos_nudged = d.xpos[plate_id].copy()  # ボディの世界座標位置（3次元ベクトル）
        dx = plate_pos_nudged[0] - plate_pos_base[0]  # X方向の変位
        dy = plate_pos_nudged[1] - plate_pos_base[1]  # Y方向の変位
        dxy = np.sqrt(dx**2 + dy**2)  # XY平面上の変位量（ユークリッド距離）

        parts.append(f"joint{jnum} +0.01 rad → plate dX={dx:+.4f} dY={dy:+.4f} dXY={dxy:.4f}")

    # 状態を復元 — 診断前の状態に完全に戻す
    d.qpos[:] = qpos_save
    d.qvel[:] = qvel_save
    d.ctrl[:] = ctrl_save
    mujoco.mj_forward(m, d)  # 関節位置から全ボディの世界座標を計算（順運動学）

    # 結果表示: どの関節がプレートを最も大きく動かすか一目で分かる
    print(f"[診断] {parts[0]}  |  {parts[1]}  |  {parts[2]}")


def ball_off_plate(d):
    """ボールがプレートから落ちたかを確認する。"""
    # ボールとプレートの世界座標の差（相対位置）を計算
    ball_rel_world = d.xpos[ball_id] - d.xpos[plate_id]
    error_x = ball_rel_world[0]  # X方向の位置ずれ
    error_y = ball_rel_world[1]  # Y方向の位置ずれ
    # プレート端（0.15m）より少し内側（0.14m）で落下判定、またはZ方向に落下
    return (abs(error_x) > 0.14 or abs(error_y) > 0.14 or ball_rel_world[2] < -0.02)


def run_simulation_step(d, pid_x, pid_y):
    """PID制御で1シミュレーションステップを実行する。(error_x, error_y, nan_detected)を返す。"""
    # 物理シミュレーションを1ステップ（0.005秒）進める
    mujoco.mj_step(model, d)

    # PID対象外の関節をホームに保持（joint1〜joint5はプレート傾きに影響しない）
    for i in [0, 1, 2, 3, 4]:  # joint1-5: アームの根元〜肘の関節
        d.ctrl[i] = home[i]  # アクチュエータi番目の制御信号
    d.ctrl[7] = 0.008  # グリッパーの指位置（プレート端を挟む）

    # センシング: 世界座標系でのボールのプレートに対する相対位置
    ball_rel_world = d.xpos[ball_id] - d.xpos[plate_id]  # ボディの世界座標位置の差
    error_x = ball_rel_world[0]  # X方向の位置ずれ（正=右にずれている）
    error_y = ball_rel_world[1]  # Y方向の位置ずれ（正=奥にずれている）

    # NaNガード — 数値が発散するとシミュレーション結果が無効になる
    if np.any(np.isnan(d.xpos[ball_id])):
        return error_x, error_y, True

    # PIDコントローラーで位置ずれから補正量を計算
    correction_x = pid_x.compute(error_x)  # X方向の補正量
    correction_y = pid_y.compute(error_y)  # Y方向の補正量

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
    # ctrl[5] → joint6（手首回転1、プレートのX方向傾き制御）— 符号が逆！
    d.ctrl[5] = home[5] - correction_x
    # ctrl[6] → joint7（手首回転2、プレートのY方向傾き制御）— 符号が逆！
    d.ctrl[6] = home[6] - correction_y

    return error_x, error_y, False


# ============================================================
# .mp4モード: 10秒間の1回実行、動画保存して終了
# ============================================================
if args.no_stream or not HAS_STREAMER:
    import mediapy

    if not args.no_stream and not HAS_STREAMER:
        print("警告: mujoco_streamerがインストールされていません。.mp4出力にフォールバックします")

    # シーンを初期状態にリセットし、関節権限を診断
    reset_scene(model, data)
    run_joint_diagnostics(model, data)

    # X方向とY方向それぞれにPIDコントローラーを作成
    pid_x = PIDController(KP, KI, KD, dt)  # X方向のボール位置ずれを制御
    pid_y = PIDController(KP, KI, KD, dt)  # Y方向のボール位置ずれを制御

    duration = 10.0  # 目標維持時間（秒）
    fps = 30         # 映像の目標フレームレート
    steps = int(duration / dt)  # 総ステップ数（10秒÷0.005秒=2000ステップ）
    # 30fpsを目標にレンダリング間隔を設定（例: timestep=0.005sなら約6ステップに1回描画）
    render_every = int(1.0 / (fps * dt))
    frames = []           # mp4保存用のフレーム画像リスト
    survival_time = duration  # ボールが落ちなければ最大10秒

    print(f"PIDゲイン: Kp={KP}, Ki={KI}, Kd={KD}")
    print(f"シミュレーション中 {duration}秒...")
    print()

    for step in range(steps):
        # PID制御付きで1ステップ実行し、誤差とNaN状態を取得
        error_x, error_y, nan_detected = run_simulation_step(data, pid_x, pid_y)

        # NaN発生時は即座に終了
        if nan_detected:
            print(f"エラー: ステップ {step} でNaN")
            survival_time = step * dt  # 実際の維持時間を記録
            break

        # ボールがプレートから落ちたら維持時間を記録して終了
        if ball_off_plate(data):
            survival_time = (step + 1) * dt
            break

        # 200ステップ（1秒）ごとに誤差と制御信号を表示
        t = (step + 1) * dt
        if step % 200 == 0:
            print(f"  t={t:.1f}s  誤差: x={error_x:+.4f} y={error_y:+.4f}  "
                  f"ctrl6={data.ctrl[5]:.3f} ctrl7={data.ctrl[6]:.3f}")

        # レンダリング間隔に達したらフレームを保存
        if step % render_every == 0:
            renderer.update_scene(data, camera=cam_id)  # カメラ視点で描画データを更新
            frames.append(renderer.render())  # 画像をリストに追加

    print()
    # 維持時間を表示 — この数値を大きくすることがワークショップの目標
    print(f"維持時間: {survival_time:.1f} 秒")

    # フレームリストをmp4動画として書き出す
    mediapy.write_video("attempt_1.mp4", frames, fps=fps)
    print(f"動画を保存しました: attempt_1.mp4 ({len(frames)} フレーム)")

# ============================================================
# 配信モード: 自動リセットループ付きライブMJPEG
# ============================================================
else:
    print(f"ポート {stream_port} でライブ配信を開始...")
    print(f"PIDゲイン: Kp={KP}, Ki={KI}, Kd={KD}")
    print(f"Ctrl+C で停止できます。\n")

    # 指定ポートでHTTPサーバーを起動し、ブラウザから映像を見られるようにする
    streamer = LiveStreamer(port=stream_port)
    streamer.start()
    # マウスで視点を動かせるフリーカメラを作成
    cam = streamer.make_free_camera(model)
    fps = 30  # 映像の目標フレームレート
    # 30fpsを目標にレンダリング間隔を設定
    render_every = int(1.0 / (fps * dt))
    attempt = 0  # 試行回数カウンター

    try:
        while True:  # 外側ループ: ボール落下のたびに自動リセット
            # 新しい試行のためにリセット
            attempt += 1
            reset_scene(model, data)
            run_joint_diagnostics(model, data)  # 手首関節の権限を診断表示

            # X方向とY方向それぞれにPIDコントローラーを新規作成（積分値もリセット）
            pid_x = PIDController(KP, KI, KD, dt)
            pid_y = PIDController(KP, KI, KD, dt)

            print(f"--- 試行 {attempt} ---")
            step = 0
            survival_time = 0.0

            while True:  # 内側ループ: 各ステップでPID制御を実行
                # PID制御付きで1ステップ実行し、誤差とNaN状態を取得
                error_x, error_y, nan_detected = run_simulation_step(data, pid_x, pid_y)
                step += 1

                # NaN発生時は試行を終了
                if nan_detected:
                    print(f"エラー: ステップ {step} でNaN")
                    survival_time = step * dt
                    break

                # ボールがプレートから落ちたら維持時間を記録して試行終了
                if ball_off_plate(data):
                    survival_time = (step) * dt
                    break

                # 200ステップ（1秒）ごとに誤差と制御信号を表示
                t = step * dt
                if step % 200 == 0:
                    print(f"  t={t:.1f}s  誤差: x={error_x:+.4f} y={error_y:+.4f}  "
                          f"ctrl6={data.ctrl[5]:.3f} ctrl7={data.ctrl[6]:.3f}")

                # フレームをブラウザに配信
                if step % render_every == 0:
                    # ブラウザからのカメラ操作（回転・ズーム）を反映
                    streamer.drain_camera_commands(model, cam, renderer.scene)
                    # 現在の物理状態をカメラ視点で描画データに変換
                    renderer.update_scene(data, camera=cam)
                    # レンダリング結果（画像）をブラウザに送信
                    streamer.update(renderer.render())

            # 維持時間を表示 — この数値を大きくすることがワークショップの目標
            print(f"維持時間: {survival_time:.1f} 秒")

            # リセット前にボールが床に落ちるのを見せる（視覚的フィードバック）
            if not nan_detected:
                max_fall = int(3.0 / dt)    # 最大3秒間の落下アニメーション
                settle = int(0.5 / dt)      # 着地後0.5秒待つ
                landed_at = None            # 着地タイミング記録用
                for fall_step in range(max_fall):
                    # 物理シミュレーションを1ステップ（0.005秒）進める
                    mujoco.mj_step(model, data)
                    # アームはホームポーズに保持（ボールだけ自由落下）
                    for i in range(7):
                        data.ctrl[i] = home[i]  # アクチュエータi番目の制御信号
                    data.ctrl[7] = 0.008  # グリッパーの指位置（プレート端を挟む）
                    # 落下中もフレームをブラウザに配信
                    if fall_step % render_every == 0:
                        streamer.drain_camera_commands(model, cam, renderer.scene)
                        renderer.update_scene(data, camera=cam)
                        streamer.update(renderer.render())
                    ball_z = data.xpos[ball_id][2]  # ボールのZ座標（高さ）
                    # ボールが地面近く（Z < 0.05m）に達したら着地と判定
                    if landed_at is None and ball_z < 0.05:
                        landed_at = fall_step
                    # 着地後0.5秒経過したら落下アニメーション終了
                    if landed_at is not None and (fall_step - landed_at) >= settle:
                        break
                    # 落下中にNaNが出たらアニメーション中断
                    if np.any(np.isnan(data.xpos[ball_id])):
                        break
            print()

    except KeyboardInterrupt:
        print("\n配信を停止しました。")
    finally:
        # 必ずサーバーを停止してポートを解放する
        streamer.stop()

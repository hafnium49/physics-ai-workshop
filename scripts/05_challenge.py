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
        self.kp = kp    # 比例ゲイン（現在の誤差に比例して補正する強さ）
        self.ki = ki    # 積分ゲイン（誤差の蓄積を補正する強さ — 定常偏差を除去）
        self.kd = kd    # 微分ゲイン（誤差の変化速度に反応する強さ — 振動を抑制）
        self.dt = dt    # シミュレーションの時間刻み（秒）
        self.integral = 0.0   # 誤差の累積値（積分項）
        self.prev_error = 0.0  # 前回の誤差（微分計算用）

    def compute(self, error):
        """誤差からPID制御出力を計算する。毎ステップ呼び出される。"""
        self.integral += error * self.dt          # 積分項: 誤差×時間刻みを累積
        derivative = (error - self.prev_error) / self.dt  # 微分項: 誤差の変化率（速度）
        # PID出力 = P項（位置補正）+ I項（蓄積補正）+ D項（速度補正）
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error  # 次回の微分計算のために今回の誤差を保存
        return output

    def reset(self):
        """試行間でコントローラーの内部状態をリセットする。"""
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

    # X軸・Y軸それぞれ独立したPIDコントローラーを作成
    pid_x = PIDController(kp, 0.0, kd, dt)  # X方向のボール位置制御
    pid_y = PIDController(kp, 0.0, kd, dt)  # Y方向のボール位置制御

    # クロージャ: 上で定義したpid_x, pid_y, homeを内部で参照し続ける関数を返す
    def controller(data, plate_id, ball_id, step, t):
        # ボールのプレート中心からの相対位置（メートル単位）
        brel = data.xpos[ball_id] - data.xpos[plate_id]
        ex, ey = brel[0], brel[1]  # X方向・Y方向の誤差（ずれ）

        # PIDで補正量を計算（ボールがずれた方向にプレートを傾ける）
        correction_x = pid_x.compute(ex)
        correction_y = pid_y.compute(ey)

        # ctrl[5]=joint6（手首回転）、ctrl[6]=joint7（手首回転）に補正を加える
        # home値にPID補正を足すことで、ホーム姿勢を基準に微調整する
        data.ctrl[5] = home[5] + correction_x
        data.ctrl[6] = home[6] + correction_y

    return controller  # この関数がシミュレーションの毎ステップで呼ばれる


# ═══════════════════════════════════════════════════════════
# 以下は編集しないでください — テスト用スタンドアロンランナー
# ═══════════════════════════════════════════════════════════

# このファイルを直接実行したときだけ以下が動く（04_survival_map.pyから読み込まれた場合は動かない）
if __name__ == "__main__":
    import os
    import sys
    # EGLレンダリング: ヘッドレス環境（画面なし）でMuJoCoを描画するための設定
    os.environ.setdefault("MUJOCO_GL", "egl")
    # 親ディレクトリをPythonパスに追加（mujoco_streamerをインポートするため）
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import argparse
    import mujoco
    from mujoco_streamer import LiveStreamer

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir)  # プロジェクトのルートディレクトリ

    # --- CLI（コマンドライン引数の解析） ---
    parser = argparse.ArgumentParser(
        description="コントローラーの10秒間簡易テスト")
    parser.add_argument("--port", type=int, default=None,
                        help="配信ポート（デフォルト: STREAM_PORT環境変数または18080）")
    args = parser.parse_args()
    # ポート優先順位: コマンドライン引数 > 環境変数 > デフォルト18080
    stream_port = args.port if args.port is not None else int(os.environ.get("STREAM_PORT", 18080))

    print("コントローラーの簡易テストを実行中。完全な維持マップを取得するには:")
    print("  python scripts/04_survival_map.py --controller scripts/05_challenge.py")
    print()

    # --- モデルの読み込み ---
    # XMLファイルからMuJoCoモデルを構築（ロボットアーム＋プレート＋ボールの定義）
    model = mujoco.MjModel.from_xml_path(
        os.path.join(_project_root, "content", "panda_ball_balance.xml"))
    dt = model.opt.timestep  # シミュレーションの時間刻み（0.005秒 = 200Hz）

    # 名前からボディIDを取得（位置追跡に使う）
    plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    # ボールの自由関節ID（6自由度: 位置3 + 回転4のクォータニオン）
    ball_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    ball_qpos_addr = model.jnt_qposadr[ball_joint_id]  # qpos配列内でのボール位置の開始インデックス
    ball_qvel_addr = model.jnt_dofadr[ball_joint_id]    # qvel配列内でのボール速度の開始インデックス

    # ロボットアームのホーム姿勢（7関節の角度、ラジアン）
    home = [0.0, -0.785, 0.0, -2.356, 1.184, 3.184, 1.158]
    joint_names = [f"joint{i}" for i in range(1, 8)]  # joint1〜joint7

    # --- データ作成とホームポーズの設定 ---
    data = mujoco.MjData(model)  # シミュレーション状態を保持するオブジェクト
    for jn, val in zip(joint_names, home):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        data.qpos[model.jnt_qposadr[jid]] = val  # 各関節の初期角度を設定
    for i, val in enumerate(home):
        data.ctrl[i] = val  # アクチュエータの目標値もホーム姿勢に設定
    data.ctrl[7] = 0.008  # グリッパーをわずかに開く（8mm）
    # 順運動学計算（関節角度→世界座標）— ボディの位置を計算
    mujoco.mj_forward(model, data)

    # ボールをプレート上に配置
    # プレート中心の真上25mm（ボール半径20mm + 余裕5mm）に置く
    data.qpos[ball_qpos_addr:ball_qpos_addr + 3] = data.xpos[plate_id] + [0, 0, 0.025]
    data.qpos[ball_qpos_addr + 3:ball_qpos_addr + 7] = [1, 0, 0, 0]  # 単位クォータニオン（回転なし）
    data.qvel[ball_qvel_addr:ball_qvel_addr + 6] = 0  # ボールの初速度をゼロに
    mujoco.mj_forward(model, data)  # 順運動学計算（関節角度→世界座標）— 配置を反映

    # --- レンダラーとストリーマー ---
    renderer = mujoco.Renderer(model, height=480, width=640)  # オフスクリーンレンダラー
    print(f"ポート {stream_port} でライブ配信を開始中...")
    streamer = LiveStreamer(port=stream_port)  # MJPEGストリーミングサーバー
    streamer.start()  # HTTPサーバーをバックグラウンドスレッドで起動
    cam = streamer.make_free_camera(model)  # モデルのデフォルト設定でフリーカメラを初期化

    # --- シミュレーションパラメータ ---
    duration = 10.0                    # テスト時間（秒）
    steps = int(duration / dt)         # 総ステップ数（10秒 / 0.005秒 = 2000ステップ）
    fps = 30                           # 配信フレームレート（秒間30フレーム）
    render_every = int(1.0 / (fps * dt))  # 何ステップごとに描画するか（≒7ステップごと）

    try:
        # 無限ループ: ボールが落ちたら自動的にリセットして再試行
        while True:
            # シーンをリセット（全関節・速度・力を初期化）
            mujoco.mj_resetData(model, data)
            for jn, val in zip(joint_names, home):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                data.qpos[model.jnt_qposadr[jid]] = val  # 関節角度をホーム姿勢に復帰
            for i, val in enumerate(home):
                data.ctrl[i] = val  # アクチュエータ目標値もホーム姿勢に
            data.ctrl[7] = 0.008  # グリッパー開度
            mujoco.mj_forward(model, data)  # 順運動学計算（関節角度→世界座標）

            # ボールをプレート上に再配置
            data.qpos[ball_qpos_addr:ball_qpos_addr + 3] = data.xpos[plate_id] + [0, 0, 0.025]
            data.qpos[ball_qpos_addr + 3:ball_qpos_addr + 7] = [1, 0, 0, 0]  # 回転なし
            data.qvel[ball_qvel_addr:ball_qvel_addr + 6] = 0  # 速度ゼロ
            mujoco.mj_forward(model, data)  # 順運動学計算（関節角度→世界座標）

            # この試行用のコントローラーを作成（PID内部状態がリセットされる）
            controller_fn = make_controller(model, dt, home)
            survival_time = duration  # ボールが落ちなければ10秒（満点）

            for step in range(steps):
                mujoco.mj_step(model, data)  # 物理シミュレーションを1ステップ進める

                # PID制御対象外の関節（joint1〜5）をホーム姿勢に固定
                for i in [0, 1, 2, 3, 4]:
                    data.ctrl[i] = home[i]
                data.ctrl[7] = 0.008  # グリッパー開度を維持

                # ユーザーが編集したコントローラーを実行（ctrl[5], ctrl[6]を更新）
                t = step * dt  # 現在のシミュレーション時刻（秒）
                controller_fn(data, plate_id, ball_id, step, t)

                # NaNチェック（シミュレーションが発散した場合の安全停止）
                if np.any(np.isnan(data.xpos[ball_id])):
                    survival_time = step * dt
                    break

                # ボール落下チェック（プレート端から14cm以上ずれたら失敗）
                brel = data.xpos[ball_id] - data.xpos[plate_id]
                # プレートサイズは15cm×15cmなので、14cmで端に到達
                if abs(brel[0]) > 0.14 or abs(brel[1]) > 0.14 or brel[2] < -0.02:
                    survival_time = (step + 1) * dt
                    break

                # 描画タイミング（毎ステップではなく30fps相当で描画して負荷軽減）
                if step % render_every == 0:
                    # ブラウザからのカメラ操作コマンドを受信して適用
                    streamer.drain_camera_commands(model, cam, renderer.scene)
                    renderer.update_scene(data, camera=cam)  # シーンを更新
                    streamer.update(renderer.render())  # フレームをブラウザに配信

            print(f"維持時間: {survival_time:.1f} 秒")
            print()

    except KeyboardInterrupt:
        print("\n停止しました。")
    finally:
        streamer.stop()  # HTTPサーバーを停止してリソースを解放

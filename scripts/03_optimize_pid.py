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

import mujoco
import numpy as np


# ---------------------------------------------------------------------------
# make_controller: 04_survival_map.py からインポート可能なPIDコントローラ
# ---------------------------------------------------------------------------
def make_controller(model, dt, home):
    """Factory: return a PID controller using the correct joints and sign.

    This is the 'working controller' that Sprint 3 demonstrates.
    Uses joint6 (ctrl[5]) for X and joint7 (ctrl[6]) for Y, positive sign.
    """
    kp = 2.0
    kd = 2.0
    prev_ex = 0.0
    prev_ey = 0.0

    def controller(data, plate_id, ball_id, step, t):
        nonlocal prev_ex, prev_ey
        brel = data.xpos[ball_id] - data.xpos[plate_id]
        ex, ey = brel[0], brel[1]
        dx = (ex - prev_ex) / dt
        dy = (ey - prev_ey) / dt
        data.ctrl[5] = home[5] + (kp * ex + kd * dx)
        data.ctrl[6] = home[6] + (kp * ey + kd * dy)
        prev_ex, prev_ey = ex, ey

    return controller


# 指定パラメータで1回のシミュレーション試行を実行する関数
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
    data = mujoco.MjData(model)   # シミュレーションの状態を格納するデータオブジェクトを新規作成
    dt = model.opt.timestep       # シミュレーション時間刻み（0.005秒 = 200Hz）

    # アームをホームポーズに設定（各関節の角度を初期値にセット）
    for jn, val in zip(joint_names, home):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)  # 名前からID番号を取得
        data.qpos[model.jnt_qposadr[jid]] = val  # 関節の位置（角度）を設定
    for i, val in enumerate(home):
        data.ctrl[i] = val  # アクチュエータi番目の制御信号をホーム角度に設定
    data.ctrl[7] = 0.008  # グリッパーの指位置（プレート端を挟む幅）
    mujoco.mj_forward(model, data)  # 関節位置から全ボディの世界座標を計算（順運動学）

    # ボールをプレート中心の真上に配置
    ba = model.jnt_qposadr[ball_joint_id]   # ボール自由関節のqpos配列内の開始アドレス
    bv = model.jnt_dofadr[ball_joint_id]    # ボール自由関節のqvel配列内の開始アドレス
    data.qpos[ba:ba + 3] = data.xpos[plate_id] + [0, 0, 0.025]  # プレート上面に配置（0.025 = ボール半径0.02m + マージン）
    data.qpos[ba + 3:ba + 7] = [1, 0, 0, 0]   # 回転なし（単位クォータニオン）
    data.qvel[bv:bv + 6] = 0                    # 速度ゼロ（静止状態）
    mujoco.mj_forward(model, data)               # 関節位置から全ボディの世界座標を計算（順運動学）

    renderer = None
    frames = []      # 映像フレームを蓄積するリスト
    cam_id = -1      # カメラID（-1はデフォルトカメラ）
    if render:
        renderer = mujoco.Renderer(model, height=480, width=640)  # オフスクリーンレンダラー作成
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side")  # サイドカメラのIDを取得

    fps = 30                               # 映像の目標フレームレート
    render_every = int(1.0 / (fps * dt))   # 何ステップごとにフレームを記録するか（= 約6〜7ステップ）
    prev_ex, prev_ey = 0.0, 0.0            # 前ステップのボール誤差（微分項の計算用）
    steps = int(duration / dt)             # 総シミュレーションステップ数

    for step in range(steps):
        mujoco.mj_step(model, data)  # 物理シミュレーションを1ステップ（0.005秒）進める

        # 全関節をホームに保持（PID補正しない関節が動かないように）
        for i, val in enumerate(home):
            data.ctrl[i] = val         # アクチュエータi番目の制御信号をホーム角度に設定
        data.ctrl[7] = 0.008           # グリッパーの指位置（プレート端を挟む）

        # ボールのプレート中心からの相対位置を計算（誤差 = 制御対象）
        brel = data.xpos[ball_id] - data.xpos[plate_id]  # ボディの世界座標位置の差
        ex, ey = brel[0], brel[1]          # X方向誤差, Y方向誤差（メートル）
        dx = (ex - prev_ex) / dt           # X方向誤差の時間微分（速度）→ D制御に使用
        dy = (ey - prev_ey) / dt           # Y方向誤差の時間微分（速度）→ D制御に使用

        # 選択された関節にPID補正を適用（P: 位置に比例、D: 速度に比例）
        # ctrl[5] = joint6（手首回転1、X方向傾き制御）
        # ctrl[6] = joint7（手首回転2、Y方向傾き制御）
        data.ctrl[joint_y_idx] = home[joint_y_idx] + sign * (kp * ey + kd * dy)
        data.ctrl[joint_x_idx] = home[joint_x_idx] + sign * (kp * ex + kd * dx)

        prev_ex, prev_ey = ex, ey  # 次ステップの微分計算用に誤差を保存

        # NaNチェック（シミュレーションが不安定で発散した場合）
        if np.any(np.isnan(data.xpos[ball_id])):
            t = step * dt
            return (t, frames) if render else t

        # ボールがプレートから落下したかチェック（0.14 = プレート端0.15mより少し内側）
        if abs(ex) > 0.14 or abs(ey) > 0.14 or brel[2] < -0.02:
            t = (step + 1) * dt  # 落下した時刻を計算
            return (t, frames) if render else t

        # レンダリング（指定フレームレートに合わせて間引き）
        if render and step % render_every == 0:
            renderer.update_scene(data, camera=cam_id)  # シーンを更新
            frames.append(renderer.render())             # フレーム画像を蓄積

    # 全ステップ完了 = ボールがduration秒間維持された
    return (duration, frames) if render else duration


if __name__ == "__main__":
    import argparse

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
    stream_port = args.port if args.port is not None else int(os.environ.get("STREAM_PORT", 18080))

    # ---------------------------------------------------------------------------
    # モデルの読み込み
    # ---------------------------------------------------------------------------
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir)

    # プレート付きPandaアームとボールが組み立て済みのモデルを読み込み
    model = mujoco.MjModel.from_xml_path(os.path.join(_project_root, "content", "panda_ball_balance.xml"))

    # 名前からID番号を取得（シミュレーション中の参照用）
    plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")

    # ホームポーズ: アームが自然な姿勢で静止する関節角度（ラジアン）
    home = [0.0, -0.785, 0.0, -2.356, 1.184, 3.184, 1.158]
    joint_names = [f"joint{i}" for i in range(1, 8)]

    try:
        # --- フェーズ1: どの関節ペアがプレート傾きを制御できるか検証する ---
    print("=" * 60)
    print("フェーズ1: 関節ペアの検証 (Kp=2, Kd=0)")
    print("=" * 60)

    # テストする関節ペアの候補（X軸制御用, Y軸制御用）
    pairings = [
        ("j6(X)+j7(Y)", 5, 6),   # 正しいペア（手首の2軸回転）
        ("j6(X)+j5(Y)", 5, 4),   # 代替ペア（動作する場合あり）
        ("j5(X)+j6(Y)", 4, 5),   # 軸が逆（制御方向を間違えた場合）
        ("j5(X)+j4(Y)", 4, 3),   # 完全に間違った関節（手首ではなく肘付近）
    ]

    # 各ペアで符号+1と-1を試す（補正方向が合わないとボールが加速して落ちる）
    for name, jx, jy in pairings:
        for sign in [+1, -1]:
            t = run_trial(jx, jy, sign, kp=2, kd=0, duration=5.0)  # 5秒間試行
            marker = " <-- 成功" if t >= 5.0 else ""
            print(f"  {name} sign={sign:+d} -> {t:.1f}s{marker}")

    # --- フェーズ2: 最良の関節ペア(j6+j7)でPIDゲインの組み合わせを網羅探索 ---
    print()
    print("=" * 60)
    print("フェーズ2: ゲイン探索 j6(X)+j7(Y), sign=+1")
    print("=" * 60)

    results = []
    # Kp（比例ゲイン）: ボール位置誤差に対する反応の強さ
    # Kd（微分ゲイン）: ボール速度に対する減衰の強さ
    for kp in [1, 2, 3, 5, 10]:
        for kd in [0, 1, 2, 5]:
            t = run_trial(5, 6, +1, kp, kd)  # 指定パラメータで10秒間の試行
            results.append((kp, kd, t))       # 結果を保存
            marker = " ***" if t >= 10.0 else ""  # 10秒完走なら強調表示
            print(f"  Kp={kp:>3d} Kd={kd:>2d} -> 維持時間: {t:.1f}秒{marker}")

    # 最も長くボールを維持できたゲインの組み合わせを選択
    best = max(results, key=lambda x: x[2])
    print(f"\n最良: Kp={best[0]}, Kd={best[1]}, 維持時間={best[2]:.1f}秒")
except KeyboardInterrupt:
    print("\nグリッド探索が中断されました。")
    raise SystemExit(0)

# --- フェーズ3: 最良結果をレンダリング（--no-renderでなければ） ---
if args.no_render:
    print("\n--no-render が指定されたため、映像出力をスキップします。")
else:
    use_stream = (not args.no_stream) and HAS_STREAMER  # ライブ配信が可能かどうか

    if use_stream:
        # ---- 最良ゲインで再実行し、ブラウザにライブMJPEG配信 ----
        print(f"\nポート {stream_port} で最良結果を配信中...")
        renderer = mujoco.Renderer(model, height=480, width=640)  # オフスクリーンレンダラー
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side")  # サイドカメラID
        streamer = LiveStreamer(port=stream_port)  # MJPEG配信サーバー作成
        streamer.start()                           # 配信開始
        cam = streamer.make_free_camera(model)     # ブラウザからカメラ操作可能にする

        data = mujoco.MjData(model)                  # 新しいシミュレーション状態
        dt = model.opt.timestep                      # 時間刻み（0.005秒）
        ba = model.jnt_qposadr[ball_joint_id]        # ボール位置のqposアドレス
        bv = model.jnt_dofadr[ball_joint_id]         # ボール速度のqvelアドレス
        fps = 30                                     # 配信フレームレート
        render_every = int(1.0 / (fps * dt))         # フレーム描画間隔（ステップ数）
        steps = int(10.0 / dt)                       # 10秒分のステップ数

        print(f"配信中 Kp={best[0]}, Kd={best[1]}（自動リセットループ）")
        print("Ctrl+C で停止できます。\n")

        try:
            while True:  # ボールが落ちたら自動的にリセットして繰り返す
                # 各イテレーションでシーンをリセット（全状態を初期化）
                mujoco.mj_resetData(model, data)
                for jn, val in zip(joint_names, home):
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                    data.qpos[model.jnt_qposadr[jid]] = val  # 関節角度をホームに
                for i, val in enumerate(home):
                    data.ctrl[i] = val  # アクチュエータ制御信号をホームに
                data.ctrl[7] = 0.008   # グリッパーの指位置（プレート端を挟む）
                mujoco.mj_forward(model, data)  # 関節位置から全ボディの世界座標を計算（順運動学）

                # ボールをプレート中心に再配置
                data.qpos[ba:ba + 3] = data.xpos[plate_id] + [0, 0, 0.025]  # プレート上面（ボール半径+マージン）
                data.qpos[ba + 3:ba + 7] = [1, 0, 0, 0]  # 回転なし（単位クォータニオン）
                data.qvel[bv:bv + 6] = 0                   # 速度ゼロ（静止）
                mujoco.mj_forward(model, data)              # 順運動学を再計算
                prev_ex, prev_ey = 0.0, 0.0                 # 微分項の初期値リセット

                for step in range(steps):
                    mujoco.mj_step(model, data)  # 物理シミュレーションを1ステップ（0.005秒）進める

                    for i, val in enumerate(home):
                        data.ctrl[i] = val       # PID補正しない関節はホーム保持
                    data.ctrl[7] = 0.008          # グリッパーの指位置

                    # ボール誤差の計算（プレート中心からの相対位置）
                    brel = data.xpos[ball_id] - data.xpos[plate_id]
                    ex, ey = brel[0], brel[1]          # X, Y方向の位置誤差
                    dx = (ex - prev_ex) / dt           # X方向の誤差変化率（微分項）
                    dy = (ey - prev_ey) / dt           # Y方向の誤差変化率（微分項）

                    # best[0]=Kp, best[1]=Kd を使ってPID補正を適用
                    data.ctrl[5] = home[5] + (best[0] * ex + best[1] * dx)  # joint6（手首回転1、X方向傾き制御）
                    data.ctrl[6] = home[6] + (best[0] * ey + best[1] * dy)  # joint7（手首回転2、Y方向傾き制御）

                    prev_ex, prev_ey = ex, ey  # 次ステップの微分計算用に保存

                    if np.any(np.isnan(data.xpos[ball_id])):  # シミュレーション発散チェック
                        break
                    if abs(ex) > 0.14 or abs(ey) > 0.14 or brel[2] < -0.02:  # プレート端を超えたら落下判定
                        break

                    # 配信用フレームをレンダリング
                    if step % render_every == 0:
                        streamer.drain_camera_commands(model, cam, renderer.scene)  # ブラウザからのカメラ操作を反映
                        renderer.update_scene(data, camera=cam)   # シーンを更新
                        streamer.update(renderer.render())         # フレームをブラウザに送信
        except KeyboardInterrupt:
            print("\n配信を停止しました。")
        finally:
            streamer.stop()  # 配信サーバーを停止

    else:
        # ---- ライブ配信が使えない場合: .mp4ファイルとして保存 ----
        import mediapy

        if not args.no_stream and not HAS_STREAMER:
            print("警告: mujoco_streamerがインストールされていません。.mp4出力にフォールバックします")

        print("\n最良結果をレンダリング中...")
        # 最良ゲインで再実行（render=Trueでフレーム画像を蓄積）
        t, frames = run_trial(5, 6, +1, best[0], best[1], render=True)
        mediapy.write_video("best_balance.mp4", frames, fps=30)  # フレームをMP4動画として保存
        print(f"動画を保存しました: best_balance.mp4 ({len(frames)} フレーム, 維持時間 {t:.1f}秒)")

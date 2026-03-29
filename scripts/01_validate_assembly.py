"""ステップ2: 統合モデル panda_ball_balance.xml の検証。

モデルを読み込み、アームをホームポーズに設定し、ボールをプレートに配置して、
MJPEGライブ配信または短い.mp4として保存します。

デフォルト:  MUJOCO_GL=egl python scripts/01_validate_assembly.py        （ライブ配信）
フォールバック: MUJOCO_GL=egl python scripts/01_validate_assembly.py --no-stream --duration 3
"""
import os
import sys
# ヘッドレス環境でOpenGLレンダリングを有効にする設定（GPUオフスクリーン描画）
os.environ.setdefault("MUJOCO_GL", "egl")
# プロジェクトルートをPythonの検索パスに追加（mujoco_streamerを見つけるため）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

import mujoco
import numpy as np

# スクリプトの場所からプロジェクトルートを計算（モデルファイルの相対パス解決用）
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)

# ---------------------------------------------------------------------------
# オプション: ストリーマーのインポート
# ---------------------------------------------------------------------------
try:
    # ブラウザでリアルタイム映像を見るためのライブ配信モジュール
    from mujoco_streamer import LiveStreamer
    HAS_STREAMER = True
except ImportError:
    # インストールされていなければmp4保存にフォールバック
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

# 配信ポート: コマンドライン引数 → 環境変数 → デフォルト18080 の優先順位で決定
stream_port = args.port if args.port is not None else int(os.environ.get("STREAM_PORT", 18080))

# ライブ配信を使うかどうかの判定（--no-streamでなく、かつモジュールが利用可能）
use_stream = (not args.no_stream) and HAS_STREAMER

# ---------------------------------------------------------------------------
# モデルの読み込み
# ---------------------------------------------------------------------------
# XMLファイルからMuJoCoモデルを構築（ロボット・プレート・ボールの物理定義すべてを含む）
model = mujoco.MjModel.from_xml_path(os.path.join(_project_root, "content", "panda_ball_balance.xml"))
# シミュレーション状態（関節角度・速度・接触力など）を保持するデータ構造を作成
data = mujoco.MjData(model)

# ボディID — 名前からID番号を取得（位置の追跡に使う）
plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")  # プレートのボディID
ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")    # ボールのボディID

# 関節IDとqposアドレス — 7つのアーム関節それぞれの位置データがdata.qposの何番目にあるかを取得
joint_names = [f"joint{i}" for i in range(1, 8)]  # joint1〜joint7の名前リスト
joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]  # 名前からID番号を取得
joint_addrs = [model.jnt_qposadr[jid] for jid in joint_ids]  # 各関節のqpos配列内のインデックス

# ホームポーズ: プレートが水平になるようチューニングされた7関節の角度（ラジアン）
# j5, j6, j7 はプレートの把持姿勢に合わせて調整済み
home = [0.0, -0.785, 0.0, -2.356, 1.184, 3.184, 1.158]

# ボールの自由関節（6自由度）のqpos/qvelアドレスを取得
# ボールは free joint なので qpos に位置(x,y,z)+姿勢(quaternion)の7要素を持つ
ball_qpos_addr = model.jnt_qposadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")  # 名前からID番号を取得
]
# qvel（速度）のアドレス — 自由関節は並進3+回転3=6自由度
ball_qvel_addr = model.jnt_dofadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")  # 名前からID番号を取得
]


def reset_scene():
    """アームをホームポーズに設定し、ボールをプレートに配置し、速度をゼロにする。"""
    # すべての状態（位置・速度・力など）を初期値に戻す
    mujoco.mj_resetData(model, data)
    # 各関節にホームポーズの角度を設定
    for addr, val in zip(joint_addrs, home):
        data.qpos[addr] = val  # 全関節の角度（ラジアン）に目標値を書き込む
    # アクチュエータの制御信号もホームポーズに合わせる（位置制御なので角度を指定）
    for i, val in enumerate(home):
        data.ctrl[i] = val  # アクチュエータi番目の制御信号
    data.ctrl[7] = 0.008  # グリッパーの指位置（プレート端を挟む）
    # 関節位置から全ボディの世界座標を計算（順運動学）— プレート位置を確定させる
    mujoco.mj_forward(model, data)

    # プレートの世界座標位置を取得し、その上にボールを配置
    plate_pos = data.xpos[plate_id].copy()  # ボディの世界座標位置（3次元ベクトル）
    # ボールの位置 = プレート中心 + Z方向に0.025m（ボール半径0.02m + マージン）
    data.qpos[ball_qpos_addr:ball_qpos_addr + 3] = plate_pos + [0, 0, 0.025]
    # ボールの姿勢を単位クォータニオン（回転なし）に設定
    data.qpos[ball_qpos_addr + 3:ball_qpos_addr + 7] = [1, 0, 0, 0]
    # ボールの速度（並進3+回転3）をすべてゼロにする
    data.qvel[ball_qvel_addr:ball_qvel_addr + 6] = 0
    # 再度順運動学を計算して、ボール配置後の正確な世界座標を更新
    mujoco.mj_forward(model, data)


# シーンを初期状態にリセット
reset_scene()

# 初期状態の位置情報を表示して、組み立てが正しいか確認
print(f"プレート位置（世界座標）: {data.xpos[plate_id]}")
print(f"ボール位置（世界座標）:  {data.xpos[ball_id]}")
# Z距離が約0.025mならボールがプレートの上に正しく載っている
print(f"ボール・プレート間Z距離: {data.xpos[ball_id][2] - data.xpos[plate_id][2]:.4f} m")

# ---------------------------------------------------------------------------
# レンダラー（サイドカメラ）
# ---------------------------------------------------------------------------
# 画像描画用レンダラーを作成（480×640ピクセル）
renderer = mujoco.Renderer(model, height=480, width=640)
# モデル内で定義された「side」カメラのIDを取得
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side")

fps = 30  # 映像の目標フレームレート
# 30fpsを目標にレンダリング間隔を設定（例: timestep=0.005sなら約6ステップに1回描画）
render_every = int(1.0 / (fps * model.opt.timestep))


def ball_on_plate_check():
    """ボールがおおよそプレートの上にある場合にTrueを返す。"""
    # ボールとプレートの世界座標の差（相対位置）を計算
    rel = data.xpos[ball_id] - data.xpos[plate_id]
    # X,Yが±0.15m以内かつZ方向に-0.02m以上ならプレート上と判定
    return abs(rel[0]) < 0.15 and abs(rel[1]) < 0.15 and rel[2] > -0.02


def print_diagnostics(step):
    """プレート・ボールの位置とプレート上の状態を表示する。"""
    # ボールのプレートに対する相対位置を計算
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
    # 指定ポートでHTTPサーバーを起動し、ブラウザから映像を見られるようにする
    streamer = LiveStreamer(port=stream_port)
    streamer.start()
    # マウスで視点を動かせるフリーカメラを作成
    cam = streamer.make_free_camera(model)
    print(f"\nライブ配信を開始しました http://localhost:{stream_port}")
    print("Ctrl+C で停止できます。\n")

    step = 0
    diag_interval = int(1.0 / model.opt.timestep)  # 約1秒ごと（200ステップ）に診断を出力
    try:
        while True:  # 外側ループ: NaN発生時のリカバリ
            while True:  # 内側ループ: シミュレーションステップ
                # 物理シミュレーションを1ステップ（0.005秒）進める
                mujoco.mj_step(model, data)

                # アームをホームに保持 — 全7関節のアクチュエータ制御信号をホームポーズ角度に設定
                for i, val in enumerate(home):
                    data.ctrl[i] = val  # アクチュエータi番目の制御信号

                # NaNチェック — 数値が発散するとシミュレーション結果が無効になる
                if np.any(np.isnan(data.xpos[ball_id])):
                    print(f"警告: ステップ {step} でNaNを検出、シーンをリセットします...")
                    break

                # レンダリング間隔に達したらフレームをブラウザに送信
                if step % render_every == 0:
                    # ブラウザからのカメラ操作（回転・ズーム）を反映
                    streamer.drain_camera_commands(model, cam, renderer.scene)
                    # 現在の物理状態をカメラ視点で描画データに変換
                    renderer.update_scene(data, camera=cam)
                    # レンダリング結果（画像）をブラウザに送信
                    streamer.update(renderer.render())

                # 毎秒の診断出力 — ボール位置・プレート位置を確認
                if step % diag_interval == 0:
                    print_diagnostics(step)

                step += 1

            # NaNからリカバリ: リセットして配信を継続
            reset_scene()
            step = 0
    except KeyboardInterrupt:
        print("\n配信を停止中...")
    finally:
        # 必ずサーバーを停止してポートを解放する
        streamer.stop()
        print("配信を停止しました。")

else:
    # ---- .mp4 フォールバックモード ----
    import mediapy

    if not args.no_stream and not HAS_STREAMER:
        print("警告: mujoco_streamerがインストールされていません。.mp4出力にフォールバックします。\n")

    duration = args.duration  # シミュレーション時間（秒）
    steps = int(duration / model.opt.timestep)  # 総ステップ数（例: 3秒÷0.005秒=600ステップ）
    diag_interval = int(1.0 / model.opt.timestep)  # 約1秒ごと（200ステップ）に診断を出力
    frames = []  # mp4保存用のフレーム画像リスト

    step = 0
    while step < steps:
        # 物理シミュレーションを1ステップ（0.005秒）進める
        mujoco.mj_step(model, data)

        # アームをホームに保持 — 全7関節のアクチュエータ制御信号をホームポーズ角度に設定
        for i, val in enumerate(home):
            data.ctrl[i] = val  # アクチュエータi番目の制御信号

        # NaNチェック — 数値発散時はシーン全体をリセットしてやり直す
        if np.any(np.isnan(data.xpos[ball_id])):
            print(f"警告: ステップ {step} でNaNを検出、シーンをリセットします...")
            reset_scene()
            step = 0
            frames.clear()  # 保存済みフレームもクリア
            continue

        # レンダリング間隔に達したらフレームを保存
        if step % render_every == 0:
            renderer.update_scene(data, camera=cam_id)  # カメラ視点で描画データを更新
            frames.append(renderer.render())  # 画像をリストに追加

        # 毎秒の診断出力 — ボール位置・プレート位置を確認
        if step % diag_interval == 0:
            print_diagnostics(step)

        step += 1

    # 最終診断 — シミュレーション終了時のボール・プレート位置を表示
    ball_rel = data.xpos[ball_id] - data.xpos[plate_id]
    print(f"\n最終プレート位置: {data.xpos[plate_id]}")
    print(f"最終ボール位置:  {data.xpos[ball_id]}")
    print(f"ボールのプレートに対する相対位置: x={ball_rel[0]:.4f} y={ball_rel[1]:.4f} z={ball_rel[2]:.4f}")
    print(f"プレート上: {ball_on_plate_check()}")

    # フレームリストをmp4動画として書き出す
    mediapy.write_video("assembly_test.mp4", frames, fps=fps)
    print(f"\n動画を保存しました: assembly_test.mp4 ({len(frames)} フレーム)")

"""維持マップ: ボールの初期位置をグリッドで走査し、PIDの頑健性を可視化する。

プレート中心からのオフセットのグリッド上にボールを配置し、
各点で10秒間のヘッドレスPID試行を実行し、維持時間の
等高線プロットを表示する。緑/黄色の領域は長く維持でき、
暗い領域はボールがすぐに落ちることを示す。

実行方法: python scripts/04_survival_map.py [--controller my_ctrl.py] [--kp 2] [--kd 0] [--grid 20]
"""
import os
import signal
import sys
import time
os.environ.setdefault("MUJOCO_GL", "egl")  # ヘッドレス描画用（GPUでオフスクリーンレンダリング）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # プロジェクトルートをインポートパスに追加


# タイムアウト保護（無限ループ防止）: signal.alarmから呼ばれるハンドラ
def _timeout_handler(signum, frame):
    raise TimeoutError("コントローラーがタイムアウトしました")

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)

import argparse
import importlib.util
import mujoco
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # GUIなし環境用のバックエンド（画像をメモリ上で生成）
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'  # 日本語フォント設定
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False  # matplotlibがない場合はプロット生成不可

try:
    from mujoco_streamer import LiveStreamer  # ブラウザへのリアルタイム映像配信ライブラリ
    HAS_STREAMER = True
except ImportError:
    HAS_STREAMER = False  # ライブ配信が使えない場合はPNG保存にフォールバック


# PID制御器クラス: 比例(P)・積分(I)・微分(D)の3項で誤差を補正する
class PIDController:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp    # 比例ゲイン: 現在の誤差に比例した補正力
        self.ki = ki    # 積分ゲイン: 誤差の累積に比例した補正力（定常偏差を除去）
        self.kd = kd    # 微分ゲイン: 誤差の変化速度に比例した減衰力
        self.dt = dt    # 時間刻み（秒）
        self.integral = 0.0    # 誤差の累積値（積分項）
        self.prev_error = 0.0  # 前ステップの誤差（微分項の計算用）

    def compute(self, error):
        """誤差を入力として、PID補正値を計算して返す"""
        self.integral += error * self.dt                       # 積分項: 誤差を時間で累積
        derivative = (error - self.prev_error) / self.dt       # 微分項: 誤差の変化率
        output = self.kp * error + self.ki * self.integral + self.kd * derivative  # P + I + D
        self.prev_error = error  # 次ステップ用に保存
        return output

    def reset(self):
        """試行ごとに積分値と前回誤差をリセット"""
        self.integral = 0.0
        self.prev_error = 0.0


# デフォルトPIDコントローラーを生成するファクトリ関数（試行ごとに新しいPIDを作成）
def make_default_pid(model, dt, home):
    """デフォルトPIDコントローラーのファクトリ。試行ごとに1回呼び出される。"""
    pid_x = PIDController(KP, 0.0, KD, dt)  # X方向（左右）のPID制御器
    pid_y = PIDController(KP, 0.0, KD, dt)  # Y方向（前後）のPID制御器
    def controller(data, plate_id, ball_id, step, t):
        brel = data.xpos[ball_id] - data.xpos[plate_id]  # ボールのプレート中心からの相対位置
        data.ctrl[5] = home[5] + pid_x.compute(brel[0])  # joint6（手首回転1、X方向傾き制御）
        data.ctrl[6] = home[6] + pid_y.compute(brel[1])  # joint7（手首回転2、Y方向傾き制御）
    return controller


# --- CLI引数 ---
parser = argparse.ArgumentParser(description="維持マップ: ボールの初期位置を走査")
parser.add_argument("--kp", type=float, default=2.0,
                    help="比例ゲイン（デフォルト: 2.0）")
parser.add_argument("--kd", type=float, default=0.0,
                    help="微分ゲイン（デフォルト: 0.0）")
parser.add_argument("--controller", type=str, default=None,
                    help="make_controller(model, dt, home)関数を持つコントローラーファイルのパス")
parser.add_argument("--grid", type=int, default=20,
                    help="グリッド解像度 NxN（デフォルト: 20）")
parser.add_argument("--port", type=int, default=None,
                    help="配信ポート（デフォルト: STREAM_PORT環境変数または18080）")
parser.add_argument("--no-stream", action="store_true",
                    help="ライブ配信を無効にし、survival_map.pngとして保存")
args = parser.parse_args()
stream_port = args.port if args.port is not None else int(os.environ.get("STREAM_PORT", 18080))  # 各参加者固有のポート番号

KP = args.kp   # 比例ゲイン（コマンドライン引数から）
KI = 0.0       # 積分ゲイン（このスクリプトでは未使用、0固定）
KD = args.kd   # 微分ゲイン（コマンドライン引数から）

# --- matplotlibの早期チェック ---
if not HAS_MATPLOTLIB:
    print("エラー: 維持マップにはmatplotlibが必要です。インストール: pip install matplotlib")
    sys.exit(1)

# --- セットアップ ---
# プレート付きPandaアームとボールが組み立て済みのモデルを読み込み
model = mujoco.MjModel.from_xml_path(os.path.join(_project_root, "content", "panda_ball_balance.xml"))
data = mujoco.MjData(model)    # シミュレーションの状態を格納するデータオブジェクト
dt = model.opt.timestep  # 0.005秒 = 200 Hz（シミュレーション時間刻み）

# 名前からID番号を取得（シミュレーション中にボディ位置を参照するため）
plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")  # プレートボディのID
ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")    # ボールボディのID

# ホームポーズ: アームが自然な姿勢で静止する関節角度（ラジアン）
home = [0.0, -0.785, 0.0, -2.356, 1.184, 3.184, 1.158]
joint_names = [f"joint{i}" for i in range(1, 8)]  # joint1〜joint7の名前リスト

# 関節ID（キャッシュ済み）: 毎回検索するのを避けるため辞書に保存
joint_ids = {}
for jn in joint_names:
    joint_ids[jn] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)  # 名前からID番号を取得

ball_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")  # ボールの自由関節ID（6自由度）
ball_qpos_addr = model.jnt_qposadr[ball_joint_id]  # ボール位置のqpos配列内の開始アドレス
ball_qvel_addr = model.jnt_dofadr[ball_joint_id]    # ボール速度のqvel配列内の開始アドレス

# --- ユーザー定義のコントローラー関数を動的に読み込み ---
if args.controller:
    ctrl_path = os.path.abspath(args.controller)  # コントローラーファイルの絶対パス
    if not os.path.exists(ctrl_path):
        print(f"エラー: コントローラーファイルが見つかりません: {args.controller}")
        sys.exit(1)
    try:
        # Pythonファイルをモジュールとして動的に読み込む（プラグイン方式）
        spec = importlib.util.spec_from_file_location("user_controller", ctrl_path)
        _mod = importlib.util.module_from_spec(spec)
        signal.signal(signal.SIGALRM, _timeout_handler)  # タイムアウト保護のシグナルハンドラ設定
        signal.alarm(5)  # 5秒以内にファイル読み込みが完了しなければタイムアウト（無限ループ防止）
        try:
            spec.loader.exec_module(_mod)  # ユーザーのPythonファイルを実行して読み込み
        finally:
            signal.alarm(0)  # タイムアウト解除
    except BaseException as e:
        print(f"エラー: {args.controller} の読み込みに失敗: {e}")
        print("このメッセージをClaudeに貼り付けてください。")
        print("デフォルトPIDにフォールバックします。\n")
        _mod = None
    # ユーザーファイルにmake_controller関数があるか確認
    if _mod and hasattr(_mod, 'make_controller'):
        make_ctrl = _mod.make_controller          # ユーザー定義コントローラーを使用
        controller_name = os.path.basename(args.controller)
    else:
        if _mod:
            print(f"エラー: {args.controller} にはmake_controller(model, dt, home)を定義する必要があります")
            print("デフォルトPIDにフォールバックします。\n")
        make_ctrl = make_default_pid              # デフォルトPIDにフォールバック
        controller_name = f"PID (Kp={KP}, Kd={KD})"
else:
    make_ctrl = make_default_pid                  # コントローラー未指定時はデフォルトPID
    controller_name = f"PID (Kp={KP}, Kd={KD})"


# 指定した初期位置(x0, y0)にボールを配置し、描画なしで1回のシミュレーション試行を実行
def run_headless_trial(x0, y0, make_ctrl_fn):
    """ボールを(x0, y0)オフセットに配置してヘッドレス試行を1回実行する。

    維持時間を秒で返す（最大10.0）。
    """
    d = mujoco.MjData(model)  # 新しいシミュレーション状態を作成（試行ごとに独立）
    # ホームポーズを設定（各関節の角度を初期値にセット）
    for jn, val in zip(joint_names, home):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)  # 名前からID番号を取得
        d.qpos[model.jnt_qposadr[jid]] = val  # 関節の位置（角度）を設定
    for i, val in enumerate(home):
        d.ctrl[i] = val  # アクチュエータi番目の制御信号をホーム角度に設定
    d.ctrl[7] = 0.008    # グリッパーの指位置（プレート端を挟む幅）
    mujoco.mj_forward(model, d)  # 関節位置から全ボディの世界座標を計算（順運動学）
    # プレート中心から(x0, y0)オフセットした位置にボールを配置
    ba = model.jnt_qposadr[ball_joint_id]  # ボール位置のqposアドレス
    bv = model.jnt_dofadr[ball_joint_id]   # ボール速度のqvelアドレス
    d.qpos[ba:ba+3] = d.xpos[plate_id] + [x0, y0, 0.025]  # ボール半径（0.02m）+ マージン分だけ上に配置
    d.qpos[ba+3:ba+7] = [1, 0, 0, 0]   # 回転なし（単位クォータニオン）
    d.qvel[bv:bv+6] = 0                 # 速度ゼロ（静止状態から開始）
    mujoco.mj_forward(model, d)          # 順運動学を再計算
    # コントローラーを生成して実行（毎回新しいPID状態で開始）
    controller_fn = make_ctrl_fn(model, dt, home)  # ファクトリ関数からコントローラーを生成
    max_steps = int(10.0 / dt)                     # 10秒分のシミュレーションステップ数
    signal.signal(signal.SIGALRM, _timeout_handler)  # タイムアウト保護（無限ループ防止）
    signal.alarm(15)  # 15秒以内に試行が終わらなければ強制終了
    try:
        for step in range(max_steps):
            mujoco.mj_step(model, d)  # 物理シミュレーションを1ステップ（0.005秒）進める
            if np.any(np.isnan(d.xpos[ball_id])):  # シミュレーション発散チェック
                return step * dt
            # joint1〜joint5はホーム保持（PID補正はjoint6, joint7のみ）
            for i in [0, 1, 2, 3, 4]:
                d.ctrl[i] = home[i]   # アクチュエータi番目の制御信号をホーム角度に設定
            d.ctrl[7] = 0.008          # グリッパーの指位置（プレート端を挟む）
            try:
                # ユーザー定義コントローラーを呼び出し（joint6, joint7の制御信号を更新）
                controller_fn(d, plate_id, ball_id, step, step * dt)
            except (BaseException, TimeoutError):
                return step * dt  # コントローラーエラー時は現在時刻を維持時間として返す
            # ボールのプレート中心からの相対位置で落下判定
            brel = d.xpos[ball_id] - d.xpos[plate_id]  # ボディの世界座標位置の差
            ex, ey = brel[0], brel[1]  # X, Y方向の位置誤差
            if abs(ex) > 0.14 or abs(ey) > 0.14 or brel[2] < -0.02:  # プレート端（0.15m）より少し内側で落下判定
                return (step + 1) * dt
    except TimeoutError:
        return step * dt  # タイムアウト時の維持時間
    finally:
        signal.alarm(0)  # タイムアウト解除
    return 10.0  # 全ステップ完了 = 10秒間ボールを維持できた


# プレート上の各位置にボールを配置し、維持時間を計測するグリッド走査
def run_survival_grid(grid_n, make_ctrl_fn):
    """grid_n x grid_nのヘッドレス試行を実行する。(xs, ys, survival_grid)を返す。"""
    xs = np.linspace(-0.12, 0.12, grid_n)  # X方向の初期位置（-120mm〜+120mm）
    ys = np.linspace(-0.12, 0.12, grid_n)  # Y方向の初期位置（-120mm〜+120mm）
    grid = np.zeros((grid_n, grid_n))       # 結果格納用の2D配列（各位置の維持時間）
    total = grid_n * grid_n                 # 総試行回数
    for i, y0 in enumerate(ys):
        for j, x0 in enumerate(xs):
            # 各グリッド位置で1回の試行を実行し、維持時間を記録
            grid[i, j] = run_headless_trial(x0, y0, make_ctrl_fn)
        done = (i + 1) * grid_n
        print(f"  進捗: {done}/{total} 試行 ({done*100//total}%)")
    return xs, ys, grid


# 各位置の維持時間を色で表現した2D地図（等高線プロット）を画像として生成
def render_survival_map(xs, ys, survival_grid, controller_name):
    """等高線プロットを(H, W, 3)のnumpy配列としてレンダリングする。"""
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)  # 640x480ピクセルの図を作成
    XX, YY = np.meshgrid(xs * 1000, ys * 1000)  # メートル→ミリメートルに変換（表示用）
    levels = np.linspace(0, 10, 21)  # 0〜10秒を21段階に分割（0.5秒刻みの等高線）
    # 等高線プロット（色で維持時間を表現）: 緑/黄=長く維持、暗い=すぐ落下
    cf = ax.contourf(XX, YY, survival_grid, levels=levels, cmap='viridis')
    fig.colorbar(cf, label='維持時間 (秒)')  # カラーバー（色と数値の対応表）
    ax.set_xlabel('初期Xオフセット (mm)')
    ax.set_ylabel('初期Yオフセット (mm)')
    ax.set_title(f'維持マップ ({controller_name})')
    ax.set_aspect('equal')  # X軸とY軸のスケールを同一に
    # プレート境界を白い破線の四角で表示（140mm = プレート半径150mmより少し内側）
    rect = plt.Rectangle((-140, -140), 280, 280, fill=False,
                         edgecolor='white', linewidth=1.5, linestyle='--')
    ax.add_patch(rect)
    # 最も維持時間が長かった位置を白い星でマーク
    best_idx = np.unravel_index(survival_grid.argmax(), survival_grid.shape)
    ax.plot(xs[best_idx[1]] * 1000, ys[best_idx[0]] * 1000, 'w*', markersize=15)
    # スコア注釈（全グリッド位置の平均維持時間 + 完全維持の割合）
    score = survival_grid.mean()                    # 全グリッド位置の平均維持時間
    perfect = int((survival_grid >= 9.9).sum())     # 10秒間ほぼ完全に維持できた位置の数
    total = survival_grid.size                      # 総グリッド点数
    annotation = f"スコア: {score:.1f} 秒\n完全維持: {perfect}/{total} ({perfect*100/total:.1f}%)"
    ax.text(0.02, 0.98, annotation, transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
    fig.tight_layout()  # レイアウト自動調整
    # matplotlibの図をnumpy配列（RGB画像）に変換
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)  # RGBA画像データ
    rgb = buf[:, :, :3].copy()  # アルファチャネルを除去してRGBのみ
    plt.close(fig)  # メモリ解放
    return rgb


# ============================================================
# メイン: グリッド走査を実行し、結果を可視化する
# ============================================================
print(f"=== 維持マップ ({args.grid}x{args.grid} グリッド, {controller_name}) ===")
print(f"維持マップを計算中...")

# まず中心位置（オフセット0,0）でコントローラーが動作するか確認
print(f"コントローラー: {controller_name}")
probe_t = run_headless_trial(0, 0, make_ctrl)  # プレート中心にボールを置いてテスト
if probe_t < 0.01:  # 即座に失敗 = コントローラーに問題がある可能性
    print(f"警告: コントローラーが直ちに失敗しました（中心位置での維持時間 {probe_t:.3f}秒）")
    print("コントローラーにバグがある可能性があります。このメッセージをClaudeに貼り付けてください。")
    if args.controller:
        print("デフォルトPIDにフォールバックします。\n")
        make_ctrl = make_default_pid
        controller_name = f"PID (Kp={KP}, Kd={KD})"

# プレート上の各位置にボールを配置し、維持時間を計測するグリッド走査を実行
xs, ys, grid = run_survival_grid(args.grid, make_ctrl)

# --- 結果集計 ---
score = grid.mean()                                          # 全グリッド位置の平均維持時間（スコア）
perfect = int((grid >= 9.9).sum())                           # 10秒間ほぼ完全に維持できた位置の数
total = args.grid ** 2                                       # 総グリッド点数
best_idx = np.unravel_index(grid.argmax(), grid.shape)       # 最長維持のグリッド位置インデックス

print(f"\n  ╔════════════════════════╗")
print(f"  ║  スコア: {score:.1f} 秒       ║")
print(f"  ╚════════════════════════╝")
print(f"  完全維持: {perfect}/{total} ({perfect*100/total:.1f}%)")
print(f"  最大維持: {grid.max():.1f}秒 オフセット ({xs[best_idx[1]]*1000:.0f}mm, {ys[best_idx[0]]*1000:.0f}mm)")

# 等高線プロット画像を生成（各位置の維持時間を色で表現した2D地図）
map_img = render_survival_map(xs, ys, grid, controller_name)

# ============================================================
# ストリーム無効モード: PNGファイルとして保存して終了
# ============================================================
if args.no_stream or not HAS_STREAMER:
    if not args.no_stream and not HAS_STREAMER:
        print("警告: mujoco_streamerがインストールされていません。PNG出力にフォールバックします")

    from PIL import Image
    Image.fromarray(map_img).save("survival_map.png")  # numpy配列をPNG画像として保存
    print(f"保存しました: survival_map.png ({args.grid}x{args.grid} グリッド)")

# ============================================================
# 配信モード: ブラウザに等高線プロットをリアルタイム表示
# ============================================================
else:
    print(f"ポート {stream_port} でライブ配信を開始中...")
    print("維持マップを表示中。Ctrl+Cで停止します。")

    streamer = LiveStreamer(port=stream_port)  # MJPEG配信サーバー作成
    streamer.start()                           # 配信開始

    try:
        while True:  # 静止画を繰り返し送信（ブラウザが接続している間表示し続ける）
            streamer.update(map_img)   # フレームをブラウザに送信
            time.sleep(1.0 / 30)       # 30FPS相当の間隔で更新
    except KeyboardInterrupt:
        print("\n配信を停止しました。")
    finally:
        streamer.stop()  # 配信サーバーを停止

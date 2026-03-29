"""
MuJoCo MJPEGライブストリーマー（インタラクティブカメラ制御付き）

MuJoCoシミュレーションのフレームをHTTP経由でMJPEGとしてブラウザに配信する。
ブラウザのマウス/タッチイベントでPOST /camera経由でカメラを制御できる。

重要: フレームのレンダリングはシミュレーションスレッドでのみ行うこと。
HTTPハンドラスレッドからrenderer.render()を呼ばないこと — OpenGLコンテキストは
スレッドセーフではない。レンダリング済みのnumpy配列をstreamer.update()に渡すこと。

使い方:
    from mujoco_streamer import LiveStreamer
    streamer = LiveStreamer()
    streamer.start()
    cam = streamer.make_free_camera(model)
    # シミュレーションループ内:
    streamer.drain_camera_commands(model, cam, renderer.scene)
    renderer.update_scene(data, camera=cam)
    streamer.update(renderer.render())
"""

import collections  # deque（固定長キュー）に使用
import io            # メモリ上のバイトストリーム（JPEG変換用）
import json          # カメラコマンドのJSON解析
import os
import threading     # Condition（スレッド間同期）に使用
from http.server import BaseHTTPRequestHandler, HTTPServer  # HTTPサーバー基盤
from socketserver import ThreadingMixIn  # マルチスレッドHTTPサーバー（複数クライアント対応）

import numpy as np
from PIL import Image  # NumPy配列→JPEG変換に使用


class _StreamState:
    """シミュレーションスレッドとHTTPスレッド間のフレーム受け渡し。
    スレッドセーフな単一スロットフレームバッファ。メモリ一定 — キューなし。"""

    def __init__(self):
        # スレッド間同期（新フレーム通知用）— HTTPスレッドが新フレームを待つ仕組み
        self._condition = threading.Condition()
        self._frame = None  # 最新フレーム（NumPy RGB配列）を1枚だけ保持

    def set_frame(self, rgb_array):
        """シミュレーションスレッドから呼ばれる: 新フレームをセットして待機中のHTTPスレッドに通知"""
        with self._condition:  # ロックを取得してスレッド安全に更新
            self._frame = rgb_array
            self._condition.notify_all()  # 待機中の全HTTPクライアントを起こす

    def get_frame(self):
        """HTTPスレッドから呼ばれる: 新フレームが届くまで最大1秒待つ"""
        with self._condition:
            self._condition.wait(timeout=1.0)  # 通知が来るかタイムアウトまでスリープ
            if self._frame is None:
                return None
            return self._frame.copy()  # コピーを返す（元フレームの上書きと競合しないように）


# マルチスレッドHTTPサーバー（複数ブラウザクライアントが同時接続可能）
class _StreamingServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True       # デーモンスレッド（Ctrl+Cで自動終了）
    allow_reuse_address = True  # ポートの即時再利用を許可（再起動時のエラー防止）


# ブラウザに配信するHTML（カメラ操作のJavaScript含む）
# MJPEGストリームの表示 + マウス/タッチでカメラ操作をPOST /cameraに送信
_HTML_PAGE = """\
<html>
<head>
<meta charset="utf-8">
<title>MuJoCo ライブ</title>
</head>
<body style="margin:0; background:#1a1a2e; display:flex; flex-direction:column;
             align-items:center; justify-content:center; height:100vh;
             font-family:system-ui; color:#eee; user-select:none;">

  <div id="container" style="position:relative; display:inline-block;">
    <!-- /stream エンドポイントからMJPEGを受信して表示 -->
    <img id="stream" src="/stream"
         style="border:2px solid #333; border-radius:8px;
                max-width:95vw; max-height:85vh; display:block;"
         draggable="false" />

    <!-- 透明なオーバーレイ: マウスイベントをキャプチャしてカメラ操作に変換 -->
    <div id="controls"
         style="position:absolute; inset:0; cursor:grab; border-radius:8px;
                touch-action:none;">
    </div>

    <!-- FPSと接続状態のインジケーター -->
    <div id="overlay"
         style="position:absolute; top:8px; right:12px; font-size:13px;
                background:rgba(0,0,0,0.6); padding:4px 10px;
                border-radius:4px; pointer-events:none;">
      <span id="status" style="color:#4ade80;">&#x25cf;</span>
      <span id="fps">-- fps</span>
    </div>
  </div>

  <p style="margin-top:12px; font-size:14px; opacity:0.6;">
    物理AIワークショップ — ライブシミュレーション
    &nbsp;|&nbsp; 左ドラッグ: 回転 &nbsp; スクロール: ズーム &nbsp; 右ドラッグ: 移動
    &nbsp;|&nbsp; R: カメラリセット
  </p>

  <script>
    // -- ストリーム & FPS計測 --
    const img = document.getElementById('stream');
    const fpsEl = document.getElementById('fps');
    const statusEl = document.getElementById('status');
    let frameCount = 0, lastTime = performance.now();
    img.onload = () => { frameCount++; };  // フレーム受信ごとにカウント
    img.onerror = () => {
      // 接続切断時: 赤インジケーター表示、1秒後に再接続を試行
      statusEl.style.color = '#f87171';
      fpsEl.textContent = '再接続中...';
      setTimeout(() => { img.src = '/stream?' + Date.now(); }, 1000);
    };
    // 1秒ごとにFPSを計算して表示
    setInterval(() => {
      const now = performance.now();
      const fps = Math.round(frameCount / ((now - lastTime) / 1000));
      if (frameCount > 0) { statusEl.style.color = '#4ade80'; fpsEl.textContent = fps + ' fps'; }
      frameCount = 0; lastTime = now;
    }, 1000);

    // -- カメラ制御の状態 --
    const ctrl = document.getElementById('controls');
    let activeButton = -1, lastX = 0, lastY = 0;  // ドラッグ中のマウスボタンと前回座標
    let pendingRotate = null, pendingPan = null, pendingZoom = 0;  // 蓄積中の操作量
    let inflight = false, lastSendTime = 0;  // サーバーへの送信制御
    const MIN_SEND_INTERVAL = 33;  // 最小送信間隔（ミリ秒）≒ 30fps相当

    // -- ポインターイベント（マウス/タッチ共通） --
    ctrl.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 && e.button !== 2) return;  // 左ボタン(0)と右ボタン(2)のみ
      activeButton = e.button;
      lastX = e.clientX; lastY = e.clientY;
      ctrl.setPointerCapture(e.pointerId);  // 要素外でもイベントを受信し続ける
      ctrl.style.cursor = 'grabbing';
      e.preventDefault();
    });
    ctrl.addEventListener('pointermove', (e) => {
      if (activeButton === -1) return;  // ドラッグ中でなければ無視
      const h = ctrl.clientHeight || 1;
      // マウス移動量を画面高さで正規化（解像度に依存しない操作量に変換）
      const dx = (e.clientX - lastX) / h;
      const dy = -(e.clientY - lastY) / h;  // Y軸反転（画面座標→3D座標）
      lastX = e.clientX; lastY = e.clientY;
      if (activeButton === 0) {  // 左ボタン: カメラ回転
        if (!pendingRotate) pendingRotate = {dx:0, dy:0};
        pendingRotate.dx += dx; pendingRotate.dy += dy;
      } else if (activeButton === 2) {  // 右ボタン: カメラ移動（パン）
        if (!pendingPan) pendingPan = {dx:0, dy:0};
        pendingPan.dx += dx; pendingPan.dy += dy;
      }
    });
    ctrl.addEventListener('pointerup', (e) => {
      activeButton = -1;
      ctrl.releasePointerCapture(e.pointerId);
      ctrl.style.cursor = 'grab';
    });
    ctrl.addEventListener('pointercancel', () => {
      activeButton = -1; ctrl.style.cursor = 'grab';
    });

    // -- スクロール -> ズーム --
    ctrl.addEventListener('wheel', (e) => {
      e.preventDefault();
      // スクロール量を-1〜+1に制限して蓄積
      pendingZoom += Math.max(-1, Math.min(1, -e.deltaY / 100));
    }, {passive: false});

    // -- コンテキストメニューの抑制（右クリックをカメラ操作に使うため） --
    ctrl.addEventListener('contextmenu', (e) => e.preventDefault());

    // -- スロットル付き送信ループ（蓄積した操作をまとめてPOST /cameraに送信） --
    function sendLoop() {
      requestAnimationFrame(sendLoop);  // 次フレームでも繰り返す
      const now = performance.now();
      // 前回送信から33ms未満、または前回の送信がまだ完了していなければスキップ
      if (now - lastSendTime < MIN_SEND_INTERVAL || inflight) return;
      const commands = [];
      if (pendingRotate) {
        commands.push({action:'rotate', dx:pendingRotate.dx, dy:pendingRotate.dy});
        pendingRotate = null;
      }
      if (pendingPan) {
        commands.push({action:'pan', dx:pendingPan.dx, dy:pendingPan.dy});
        pendingPan = null;
      }
      if (Math.abs(pendingZoom) > 0.001) {
        commands.push({action:'zoom', dx:0, dy:pendingZoom});
        pendingZoom = 0;
      }
      if (commands.length === 0) return;  // 操作なしならHTTPリクエストを送らない
      inflight = true; lastSendTime = now;
      // POST /camera にJSON形式でカメラコマンドを送信
      fetch('/camera', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({commands})
      }).then(() => { inflight = false; }).catch(() => { inflight = false; });
    }
    requestAnimationFrame(sendLoop);  // 送信ループを開始

    // -- キーボード: Rでカメラリセット --
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === 'r' || e.key === 'R') {
        fetch('/camera', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({commands:[{action:'reset'}]})
        });
      }
    });
  </script>
</body>
</html>
""".encode("utf-8")  # バイト列に変換（HTTPレスポンスとして送信するため）

# MJPEGフレーム間の区切り文字列（ブラウザがフレーム境界を認識するための目印）
BOUNDARY = b"FRAME"


# 公開API（start/stop/update/make_free_camera/drain_camera_commands）
class LiveStreamer:
    """
    MuJoCoシミュレーション用MJPEGライブストリーマー（インタラクティブカメラ付き）。

    公開API:
        streamer = LiveStreamer()
        streamer.start()
        cam = streamer.make_free_camera(model)
        # ループ内:
        streamer.drain_camera_commands(model, cam, renderer.scene)
        renderer.update_scene(data, camera=cam)
        streamer.update(renderer.render())
        # クリーンアップ:
        streamer.stop()
    """

    def __init__(self, port=None):
        # ポート未指定時は環境変数STREAM_PORTを参照（参加者ごとに異なるポート）
        if port is None:
            port = int(os.environ.get("STREAM_PORT", 18080))
        self._port = port
        self._stream_state = _StreamState()  # シミュレーション→HTTPスレッドへのフレーム受け渡し
        self._running = False
        self._server = None
        self._server_thread = None
        # ブラウザ→シミュレーションスレッドへのコマンドキュー（最大256件で古いものは自動破棄）
        self._camera_commands = collections.deque(maxlen=256)
        self._initial_cam_state = None  # make_free_cameraでリセット用に設定

    def make_free_camera(self, model, named="side"):
        """モデルのデフォルト設定でフリーカメラを初期化する。

        引数:
            model: MjModelインスタンス
            named: 近似するXMLカメラの名前（参照用のみ;
                   フリーカメラはmodel.statで初期ポーズを設定）

        戻り値:
            mjCAMERA_FREEに設定されたmujoco.MjvCamera
        """
        import mujoco
        cam = mujoco.MjvCamera()  # カメラオブジェクトを作成
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE  # フリーカメラモード（マウスで自由に操作可能）
        # モデルのデフォルト設定でフリーカメラを初期化（方位角・仰角・距離・注視点）
        mujoco.mjv_defaultFreeCamera(model, cam)
        # Rキーで元に戻せるよう、初期状態を保存
        self._initial_cam_state = {
            "azimuth": cam.azimuth,      # 水平方向の回転角（度）
            "elevation": cam.elevation,  # 垂直方向の回転角（度）
            "distance": cam.distance,    # 注視点からの距離（メートル）
            "lookat": cam.lookat.copy(), # カメラが見ている点の3D座標
        }
        return cam

    def drain_camera_commands(self, model, cam, scene):
        """ブラウザからの保留中のカメラコマンドをすべて適用する。

        renderer.update_scene()の前にシミュレーションスレッドから呼び出すこと。
        保留中のコマンドがなくても安全に呼び出せる。

        引数:
            model: MjModelインスタンス
            cam: MjvCameraインスタンス（mjCAMERA_FREEである必要あり）
            scene: MjvSceneインスタンス（renderer.scene）
        """
        import mujoco
        # キューに溜まったコマンドを全て処理（ブラウザ→シミュレーションスレッドへの橋渡し）
        while self._camera_commands:
            try:
                cmd = self._camera_commands.popleft()  # キューの先頭から取り出す
            except IndexError:
                break  # 別スレッドが先に取り出した場合の安全策
            action = cmd.get("action")
            dx = cmd.get("dx", 0.0)  # マウス移動量（正規化済み）
            dy = cmd.get("dy", 0.0)
            if action == "rotate":
                # カメラを移動（回転）— 左ドラッグに対応
                mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ROTATE_V, dx, dy, scene, cam)
            elif action == "pan":
                # カメラを移動（パン）— 右ドラッグに対応
                mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_MOVE_V, dx, dy, scene, cam)
            elif action == "zoom":
                # カメラを移動（ズーム）— スクロールに対応
                mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, dy, scene, cam)
            elif action == "reset" and self._initial_cam_state:
                # Rキー: カメラを初期位置に戻す
                cam.azimuth = self._initial_cam_state["azimuth"]
                cam.elevation = self._initial_cam_state["elevation"]
                cam.distance = self._initial_cam_state["distance"]
                cam.lookat[:] = self._initial_cam_state["lookat"]

    def start(self):
        """デーモンスレッドでHTTPサーバーを起動する。"""
        self._running = True

        # 内部クラスからアクセスするためにローカル変数に参照を保存
        stream_state = self._stream_state      # フレームバッファ
        camera_commands = self._camera_commands  # カメラコマンドキュー
        running_ref = self                       # サーバー停止フラグの参照

        # HTTPリクエストハンドラ（各リクエストごとにインスタンスが作られる）
        class _MJPEGHandler(BaseHTTPRequestHandler):

            def do_GET(self):
                # GET "/" — HTMLページを返す（ブラウザで最初にアクセスした時）
                if self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(_HTML_PAGE)))
                    self.end_headers()
                    self.wfile.write(_HTML_PAGE)  # カメラ操作JS付きのHTMLを送信

                # GET "/stream" — MJPEGストリームを配信（無限ループ）
                elif self.path.startswith("/stream"):
                    self.send_response(200)
                    # Motion JPEG: 連続JPEGフレームのHTTPストリーム
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=FRAME",
                    )
                    self.send_header("Cache-Control", "no-cache, no-store")  # キャッシュ無効化
                    self.end_headers()

                    try:
                        # ストリーミングループ: サーバー停止まで新フレームを送り続ける
                        while running_ref._running:
                            frame = stream_state.get_frame()  # 新フレームを待つ（最大1秒）
                            if frame is None:
                                continue  # タイムアウト — 再度待つ

                            # NumPy配列をPIL画像に変換 → JPEG圧縮
                            buf = io.BytesIO()
                            Image.fromarray(frame).save(
                                buf, format="JPEG", quality=80  # 品質80%（速度と画質のバランス）
                            )
                            jpeg_bytes = buf.getvalue()

                            # MJPEGプロトコル: 区切り文字列 → ヘッダー → JPEG本体
                            self.wfile.write(b"--" + BOUNDARY + b"\r\n")
                            self.wfile.write(
                                b"Content-Type: image/jpeg\r\n"
                            )
                            self.wfile.write(
                                f"Content-Length: {len(jpeg_bytes)}\r\n".encode()
                            )
                            self.wfile.write(b"\r\n")
                            self.wfile.write(jpeg_bytes)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()  # 即座にクライアントに送出
                    except (BrokenPipeError, ConnectionResetError):
                        pass  # ブラウザが閉じられた場合は静かに終了

                else:
                    self.send_error(404)

            def do_POST(self):
                # POST "/camera" — ブラウザからのカメラ操作コマンドを受信
                if self.path == "/camera":
                    try:
                        length = int(self.headers.get("Content-Length", 0))
                        body = json.loads(self.rfile.read(length))  # JSONを解析
                        # コマンドをキューに追加（シミュレーションスレッドが後で処理）
                        for cmd in body.get("commands", []):
                            camera_commands.append(cmd)
                    except (json.JSONDecodeError, ValueError):
                        pass  # 不正なJSONは無視
                    self.send_response(204)  # 204 No Content（本文なしの成功応答）
                    self.end_headers()
                else:
                    self.send_error(404)

            def log_message(self, format, *args):
                return  # HTTPログを抑制（コンソールが見づらくなるため）

        try:
            # 全ネットワークインターフェース(0.0.0.0)でHTTPサーバーを起動
            self._server = _StreamingServer(("0.0.0.0", self._port), _MJPEGHandler)
        except OSError:
            print(f"\nエラー: ポート {self._port} は既に使用中です。")
            print("先に他のスクリプトを停止してください（Ctrl+C）、その後再実行してください。")
            raise SystemExit(1)
        # デーモンスレッド（Ctrl+Cで自動終了）でサーバーを実行
        self._server_thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._server_thread.start()
        print(
            f"MuJoCo配信中 — ブラウザで開いてください: http://localhost:{self._port}"
        )

    def update(self, rgb_array):
        """新しいフレームをプッシュする。高速でノンブロッキング。
        シミュレーションループ内で毎フレーム呼び出す。"""
        self._stream_state.set_frame(rgb_array)  # 最新フレームをセット → 待機中のHTTPスレッドに通知

    def stop(self):
        """サーバーをクリーンにシャットダウンする。"""
        self._running = False  # ストリーミングループの終了フラグ
        # get_frame()で待機中のHTTPスレッドを起こす（タイムアウトを待たずに即座に終了させる）
        with self._stream_state._condition:
            self._stream_state._condition.notify_all()
        if self._server is not None:
            self._server.shutdown()  # HTTPサーバーを停止

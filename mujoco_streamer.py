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

import collections
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import numpy as np
from PIL import Image


class _StreamState:
    """スレッドセーフな単一スロットフレームバッファ。メモリ一定 — キューなし。"""

    def __init__(self):
        self._condition = threading.Condition()
        self._frame = None

    def set_frame(self, rgb_array):
        with self._condition:
            self._frame = rgb_array
            self._condition.notify_all()

    def get_frame(self):
        with self._condition:
            self._condition.wait(timeout=1.0)
            if self._frame is None:
                return None
            return self._frame.copy()


class _StreamingServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


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
    <img id="stream" src="/stream"
         style="border:2px solid #333; border-radius:8px;
                max-width:95vw; max-height:85vh; display:block;"
         draggable="false" />

    <div id="controls"
         style="position:absolute; inset:0; cursor:grab; border-radius:8px;
                touch-action:none;">
    </div>

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
    // -- ストリーム & FPS --
    const img = document.getElementById('stream');
    const fpsEl = document.getElementById('fps');
    const statusEl = document.getElementById('status');
    let frameCount = 0, lastTime = performance.now();
    img.onload = () => { frameCount++; };
    img.onerror = () => {
      statusEl.style.color = '#f87171';
      fpsEl.textContent = '再接続中...';
      setTimeout(() => { img.src = '/stream?' + Date.now(); }, 1000);
    };
    setInterval(() => {
      const now = performance.now();
      const fps = Math.round(frameCount / ((now - lastTime) / 1000));
      if (frameCount > 0) { statusEl.style.color = '#4ade80'; fpsEl.textContent = fps + ' fps'; }
      frameCount = 0; lastTime = now;
    }, 1000);

    // -- カメラ制御の状態 --
    const ctrl = document.getElementById('controls');
    let activeButton = -1, lastX = 0, lastY = 0;
    let pendingRotate = null, pendingPan = null, pendingZoom = 0;
    let inflight = false, lastSendTime = 0;
    const MIN_SEND_INTERVAL = 33;

    // -- ポインターイベント --
    ctrl.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 && e.button !== 2) return;
      activeButton = e.button;
      lastX = e.clientX; lastY = e.clientY;
      ctrl.setPointerCapture(e.pointerId);
      ctrl.style.cursor = 'grabbing';
      e.preventDefault();
    });
    ctrl.addEventListener('pointermove', (e) => {
      if (activeButton === -1) return;
      const h = ctrl.clientHeight || 1;
      const dx = (e.clientX - lastX) / h;
      const dy = -(e.clientY - lastY) / h;
      lastX = e.clientX; lastY = e.clientY;
      if (activeButton === 0) {
        if (!pendingRotate) pendingRotate = {dx:0, dy:0};
        pendingRotate.dx += dx; pendingRotate.dy += dy;
      } else if (activeButton === 2) {
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
      pendingZoom += Math.max(-1, Math.min(1, -e.deltaY / 100));
    }, {passive: false});

    // -- コンテキストメニューの抑制 --
    ctrl.addEventListener('contextmenu', (e) => e.preventDefault());

    // -- スロットル付き送信ループ --
    function sendLoop() {
      requestAnimationFrame(sendLoop);
      const now = performance.now();
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
      if (commands.length === 0) return;
      inflight = true; lastSendTime = now;
      fetch('/camera', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({commands})
      }).then(() => { inflight = false; }).catch(() => { inflight = false; });
    }
    requestAnimationFrame(sendLoop);

    // -- キーボード: Rでリセット --
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
""".encode("utf-8")

BOUNDARY = b"FRAME"


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
        if port is None:
            port = int(os.environ.get("STREAM_PORT", 18080))
        self._port = port
        self._stream_state = _StreamState()
        self._running = False
        self._server = None
        self._server_thread = None
        self._camera_commands = collections.deque(maxlen=256)
        self._initial_cam_state = None  # make_free_cameraでリセット用に設定

    def make_free_camera(self, model, named="side"):
        """モデルのデフォルトから初期化された永続的なフリーカメラを作成する。

        引数:
            model: MjModelインスタンス
            named: 近似するXMLカメラの名前（参照用のみ;
                   フリーカメラはmodel.statで初期ポーズを設定）

        戻り値:
            mjCAMERA_FREEに設定されたmujoco.MjvCamera
        """
        import mujoco
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        mujoco.mjv_defaultFreeCamera(model, cam)
        # リセット用に初期状態を保存
        self._initial_cam_state = {
            "azimuth": cam.azimuth,
            "elevation": cam.elevation,
            "distance": cam.distance,
            "lookat": cam.lookat.copy(),
        }
        return cam

    def drain_camera_commands(self, model, cam, scene):
        """ブラウザからの保留中のカメラコマンドを適用する。

        renderer.update_scene()の前にシミュレーションスレッドから呼び出すこと。
        保留中のコマンドがなくても安全に呼び出せる。

        引数:
            model: MjModelインスタンス
            cam: MjvCameraインスタンス（mjCAMERA_FREEである必要あり）
            scene: MjvSceneインスタンス（renderer.scene）
        """
        import mujoco
        while self._camera_commands:
            try:
                cmd = self._camera_commands.popleft()
            except IndexError:
                break
            action = cmd.get("action")
            dx = cmd.get("dx", 0.0)
            dy = cmd.get("dy", 0.0)
            if action == "rotate":
                mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ROTATE_V, dx, dy, scene, cam)
            elif action == "pan":
                mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_MOVE_V, dx, dy, scene, cam)
            elif action == "zoom":
                mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, dy, scene, cam)
            elif action == "reset" and self._initial_cam_state:
                cam.azimuth = self._initial_cam_state["azimuth"]
                cam.elevation = self._initial_cam_state["elevation"]
                cam.distance = self._initial_cam_state["distance"]
                cam.lookat[:] = self._initial_cam_state["lookat"]

    def start(self):
        """デーモンスレッドでHTTPサーバーを起動する。"""
        self._running = True

        stream_state = self._stream_state
        camera_commands = self._camera_commands
        running_ref = self

        class _MJPEGHandler(BaseHTTPRequestHandler):

            def do_GET(self):
                if self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(_HTML_PAGE)))
                    self.end_headers()
                    self.wfile.write(_HTML_PAGE)

                elif self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=FRAME",
                    )
                    self.send_header("Cache-Control", "no-cache, no-store")
                    self.end_headers()

                    try:
                        while running_ref._running:
                            frame = stream_state.get_frame()
                            if frame is None:
                                continue

                            buf = io.BytesIO()
                            Image.fromarray(frame).save(
                                buf, format="JPEG", quality=80
                            )
                            jpeg_bytes = buf.getvalue()

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
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass

                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path == "/camera":
                    try:
                        length = int(self.headers.get("Content-Length", 0))
                        body = json.loads(self.rfile.read(length))
                        for cmd in body.get("commands", []):
                            camera_commands.append(cmd)
                    except (json.JSONDecodeError, ValueError):
                        pass
                    self.send_response(204)
                    self.end_headers()
                else:
                    self.send_error(404)

            def log_message(self, format, *args):
                return

        try:
            self._server = _StreamingServer(("0.0.0.0", self._port), _MJPEGHandler)
        except OSError:
            print(f"\nエラー: ポート {self._port} は既に使用中です。")
            print("先に他のスクリプトを停止してください（Ctrl+C）、その後再実行してください。")
            raise SystemExit(1)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._server_thread.start()
        print(
            f"MuJoCo配信中 — ブラウザで開いてください: http://localhost:{self._port}"
        )

    def update(self, rgb_array):
        """新しいフレームをプッシュする。高速でノンブロッキング。"""
        self._stream_state.set_frame(rgb_array)

    def stop(self):
        """サーバーをクリーンにシャットダウンする。"""
        self._running = False
        # get_frame()で待機中のスレッドを起こす
        with self._stream_state._condition:
            self._stream_state._condition.notify_all()
        if self._server is not None:
            self._server.shutdown()

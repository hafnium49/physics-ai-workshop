"""
MuJoCo MJPEG Live Streamer

Streams MuJoCo simulation frames as MJPEG over HTTP for browser viewing.

IMPORTANT: Render frames on the simulation thread only. Do not call
renderer.render() from the HTTP handler thread — OpenGL contexts are
not thread-safe. Pass the already-rendered numpy array to streamer.update().

Usage (3 lines):
    from mujoco_streamer import LiveStreamer
    streamer = LiveStreamer(port=8080)
    streamer.start()
    # In simulation loop: streamer.update(renderer.render())
"""

import io
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import numpy as np
from PIL import Image


class _StreamState:
    """Thread-safe single-slot frame buffer. Constant memory — no queue."""

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


_HTML_PAGE = b"""\
<html>
<head><title>MuJoCo Live</title></head>
<body style="margin:0; background:#1a1a2e; display:flex; flex-direction:column; align-items:center; justify-content:center; height:100vh; font-family:system-ui; color:#eee;">
  <div style="position:relative; display:inline-block;">
    <img id="stream" src="/stream" style="border:2px solid #333; border-radius:8px; max-width:95vw; max-height:85vh;" />
    <div id="overlay" style="position:absolute; top:8px; right:12px; font-size:13px; background:rgba(0,0,0,0.6); padding:4px 10px; border-radius:4px;">
      <span id="status" style="color:#4ade80;">\xe2\x97\x8f</span> <span id="fps">-- fps</span>
    </div>
  </div>
  <p style="margin-top:12px; font-size:14px; opacity:0.6;">Physics-AI Workshop \xe2\x80\x94 Live Simulation</p>
  <script>
    const img = document.getElementById('stream');
    const fpsEl = document.getElementById('fps');
    const statusEl = document.getElementById('status');
    let frames = 0, lastTime = performance.now();
    img.onload = () => { frames++; };
    img.onerror = () => {
      statusEl.style.color = '#f87171';
      fpsEl.textContent = 'reconnecting...';
      setTimeout(() => { img.src = '/stream?' + Date.now(); }, 1000);
    };
    setInterval(() => {
      const now = performance.now();
      const fps = Math.round(frames / ((now - lastTime) / 1000));
      if (frames > 0) { statusEl.style.color = '#4ade80'; fpsEl.textContent = fps + ' fps'; }
      frames = 0; lastTime = now;
    }, 1000);
  </script>
</body>
</html>
"""

BOUNDARY = b"FRAME"


class LiveStreamer:
    """
    MJPEG live streamer for MuJoCo simulations.

    Public API:
        streamer = LiveStreamer(port=8080)
        streamer.start()
        streamer.update(rgb_array)   # call from simulation loop
        streamer.stop()              # clean shutdown
    """

    def __init__(self, port=8080):
        self._port = port
        self._stream_state = _StreamState()
        self._running = False
        self._server = None
        self._server_thread = None

    def start(self):
        """Start the HTTP server in a daemon thread."""
        self._running = True

        stream_state = self._stream_state
        running_ref = self  # closure over instance for _running check

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

            def log_message(self, format, *args):
                return

        self._server = _StreamingServer(("0.0.0.0", self._port), _MJPEGHandler)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._server_thread.start()
        print(
            f"MuJoCo streamer running — open http://localhost:{self._port} in your browser"
        )

    def update(self, rgb_array):
        """Push a new frame. Fast and non-blocking."""
        self._stream_state.set_frame(rgb_array)

    def stop(self):
        """Shut down the server cleanly."""
        self._running = False
        # Wake any threads waiting in get_frame()
        with self._stream_state._condition:
            self._stream_state._condition.notify_all()
        if self._server is not None:
            self._server.shutdown()

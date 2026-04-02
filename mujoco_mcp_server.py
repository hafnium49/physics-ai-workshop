"""FastMCP server for Physics-AI Workshop — wraps MuJoCo simulation scripts.

Two execution modes:
  - Streaming (Popen): validate_assembly, run_pid_controller
    Background process with 5-minute auto-timeout. Returns immediately
    with streaming URL. No --no-stream flag.
  - Evaluation (subprocess.run): evaluate_controller, quick_test_controller
    Blocks until completion, returns scores. Uses --no-stream flag.
"""

import atexit
import http.client
import os
import re
import socket
import subprocess
import threading
import time
from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP("physics-workshop")

WORKSHOP_DIR = Path(__file__).parent
SCRIPTS_DIR = WORKSHOP_DIR / "scripts"
STREAM_PORT = int(os.environ.get("STREAM_PORT", "18080"))
MUJOCO_GL = os.environ.get("MUJOCO_GL", "osmesa")
MAX_SIM_DURATION = 300  # 5 minutes auto-timeout

# Safe environment for subprocess execution
# Prepend the venv bin/ to PATH so subprocess finds the correct python with mujoco installed
_VENV_BIN = str(WORKSHOP_DIR / ".venv" / "bin")
SAFE_ENV = {
    "PATH": f"{_VENV_BIN}:{os.environ.get('PATH', '/usr/bin:/bin:/usr/local/bin')}",
    "HOME": "/tmp",
    "MUJOCO_GL": MUJOCO_GL,
    "PYTHONPATH": str(WORKSHOP_DIR),
    "STREAM_PORT": str(STREAM_PORT),
}

# ── Active simulation state (thread-safe) ────────────────────────────
_active_sim: subprocess.Popen | None = None
_auto_stop_timer: threading.Timer | None = None
_sim_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────

def _wait_for_port(port: int, timeout: float = 20.0, proc: subprocess.Popen | None = None) -> bool:
    """Poll until the streamer port is accepting connections.
    If proc is given, fail fast if the process dies during startup."""
    deadline = time.monotonic() + timeout
    interval = 0.2
    while time.monotonic() < deadline:
        if proc and proc.poll() is not None:
            return False  # process died, fail immediately
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(interval)
            interval = min(interval * 1.5, 1.0)
    return False


def _wait_for_port_free(port: int, timeout: float = 5.0) -> bool:
    """Poll until the port stops accepting connections (released after kill)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                pass  # still open
        except (ConnectionRefusedError, OSError):
            return True  # port is free
        time.sleep(0.2)
    return False


def _wait_for_snapshot(port: int, timeout: float = 10.0) -> bool:
    """Poll /snapshot until it returns 200 (first frame rendered)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            conn.request("GET", "/snapshot")
            resp = conn.getresponse()
            conn.close()
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def _stop_active_sim():
    """Stop the currently running simulation. Thread-safe, zombie-safe."""
    global _active_sim, _auto_stop_timer
    with _sim_lock:
        if _auto_stop_timer:
            _auto_stop_timer.cancel()
            _auto_stop_timer = None
        proc = _active_sim
        _active_sim = None
    # terminate/wait outside lock to avoid holding it during slow I/O
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)  # reap zombie


atexit.register(_stop_active_sim)


def _start_streaming_script(script_name: str, args: list[str] | None = None) -> dict:
    """Start a workshop script in streaming mode (background, 5-min auto-timeout).

    Uses subprocess.Popen — NO --no-stream flag. The script enters its
    infinite streaming loop and serves MJPEG on STREAM_PORT.

    stdout/stderr go to DEVNULL because:
      - Primary output is the MJPEG stream, not stdout.
      - PIPE would deadlock: the 64KB buffer fills in seconds with
        per-frame print statements. Nobody reads the pipe.
    """
    global _active_sim, _auto_stop_timer

    # Kill any previous simulation first
    _stop_active_sim()
    # Wait for port to be released before starting new process
    _wait_for_port_free(STREAM_PORT, timeout=5.0)

    script_path = SCRIPTS_DIR / script_name
    cmd = ["python", str(script_path)]
    if args:
        cmd.extend(args)
    # No --no-stream: script enters infinite streaming loop
    cmd.extend(["--port", str(STREAM_PORT)])

    # Capture stderr to a fixed log file for diagnosis
    _stderr_log = Path("/tmp/mujoco_sim_stderr.log")
    err_file = open(_stderr_log, "w")

    with _sim_lock:
        _active_sim = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=err_file,
            env=SAFE_ENV,
            cwd=str(WORKSHOP_DIR),
        )

        # Auto-kill after MAX_SIM_DURATION (cancelled if replaced or manually stopped)
        _auto_stop_timer = threading.Timer(MAX_SIM_DURATION, _stop_active_sim)
        _auto_stop_timer.daemon = True
        _auto_stop_timer.start()

    # Wait for streamer port — fail fast if process dies
    proc_ref = _active_sim
    if not _wait_for_port(STREAM_PORT, timeout=20.0, proc=proc_ref):
        # Read stderr for diagnosis
        err_file.flush()
        try:
            err_text = _stderr_log.read_text()[-500:]
        except Exception:
            err_text = ""

        with _sim_lock:
            if _active_sim and _active_sim.poll() is not None:
                rc = _active_sim.returncode
                _stop_active_sim()
                return {
                    "success": False,
                    "stderr": f"シミュレーション起動失敗 (exit={rc}): {err_text}",
                    "streaming": False,
                }
        return {
            "success": False,
            "stderr": f"ストリーマー起動タイムアウト（20秒）: {err_text}",
            "streaming": False,
        }

    # Wait for first frame to be rendered (snapshot returns 200, not 503)
    _wait_for_snapshot(STREAM_PORT, timeout=10.0)

    return {
        "success": True,
        "streaming": True,
        "streaming_url": f"http://0.0.0.0:{STREAM_PORT}/",
        "timeout_seconds": MAX_SIM_DURATION,
        "message": f"シミュレーション実行中（{MAX_SIM_DURATION // 60}分後に自動停止）。",
    }


def _run_script(script_name: str, args: list[str] | None = None, timeout: int = 120) -> dict:
    """Run a workshop script synchronously and capture output.

    Used by evaluation tools that need --no-stream to block until
    completion and return scores.
    """
    script_path = SCRIPTS_DIR / script_name
    cmd = ["python", str(script_path)]
    if args:
        cmd.extend(args)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            env=SAFE_ENV, cwd=str(WORKSHOP_DIR),
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[-2000:],  # Last 2KB
            "stderr": result.stderr[-1000:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": f"タイムアウト: {timeout}秒超過"}


# ── Streaming tools (Popen, no --no-stream) ───────────────────────────

@mcp.tool()
def validate_assembly(duration: float = 5.0) -> dict:
    """Sprint 1: ロボットアームとプレートの構成を確認する（ライブストリーム）。
    シミュレーションをバックグラウンドで起動し、ストリーミングURLを返す。

    Args:
        duration: シミュレーション時間（秒）。デフォルト: 5.0
    """
    return _start_streaming_script("01_validate_assembly.py")


@mcp.tool()
def run_pid_controller(
    kp: float = 50.0,
    kd: float = 10.0,
    script: str = "02_pid_baseline.py",
) -> dict:
    """Sprint 2-3: PIDコントローラでボールバランスを実行する（ライブストリーム）。
    シミュレーションをバックグラウンドで起動し、ストリーミングURLを返す。

    Args:
        kp: 比例ゲイン。デフォルト: 50.0（02_pid_baseline用）。03_optimize_pidでは無視される。
        kd: 微分ゲイン。デフォルト: 10.0（02_pid_baseline用）。03_optimize_pidでは無視される。
        script: 使用するスクリプト。"02_pid_baseline.py" or "03_optimize_pid.py"
    """
    args = []
    if script == "02_pid_baseline.py":
        args.extend(["--kp", str(kp), "--kd", str(kd)])
    return _start_streaming_script(script, args)


# ── Stop tool ─────────────────────────────────────────────────────────

@mcp.tool()
def stop_simulation() -> dict:
    """実行中のシミュレーションを停止する。
    バックグラウンドで動いているシミュレーションプロセスを終了する。
    """
    with _sim_lock:
        was_running = _active_sim is not None and _active_sim.poll() is None
    _stop_active_sim()
    if was_running:
        return {"success": True, "message": "シミュレーションを停止しました。"}
    return {"success": True, "message": "実行中のシミュレーションはありませんでした。"}


# ── Evaluation tools (subprocess.run, --no-stream) ────────────────────

@mcp.tool()
def evaluate_controller(controller_code: str, grid_size: int = 20) -> dict:
    """Sprint 4-5: コントローラをグリッド評価し、スコアと維持マップを生成する。

    Args:
        controller_code: make_controller() 関数を含むPythonコード
        grid_size: 評価グリッドサイズ。デフォルト: 20（20x20 = 400試行）
    """
    from sandbox import validate_controller_code, execute_controller_safely

    valid, error = validate_controller_code(controller_code)
    if not valid:
        return {"success": False, "error": error, "controller_score": 0.0}

    result = execute_controller_safely(
        controller_code=controller_code,
        script_path=str(SCRIPTS_DIR / "04_survival_map.py"),
        script_args=["--no-stream", "--grid", str(grid_size)],
        timeout=300,
        cwd=str(WORKSHOP_DIR),
    )

    # Parse controller score from stdout
    # Pattern: "スコア: 3.8 秒"
    score = 0.0
    perfect_count = 0
    total_trials = grid_size * grid_size
    heatmap_path = ""

    for line in result.get("stdout", "").split("\n"):
        score_match = re.search(r'スコア[:\s]+([0-9.]+)\s*秒', line)
        if score_match:
            score = float(score_match.group(1))
        # Pattern: "完全維持: 156/400 (39.0%)"
        perfect_match = re.search(r'完全維持[:\s]+(\d+)/(\d+)', line)
        if perfect_match:
            perfect_count = int(perfect_match.group(1))
            total_trials = int(perfect_match.group(2))
        # Pattern: "保存しました: survival_map.png"
        heatmap_match = re.search(r'保存しました[:\s]+(survival_map\.png)', line)
        if heatmap_match:
            heatmap_path = heatmap_match.group(1)

    result["controller_score"] = score
    result["perfect_count"] = perfect_count
    result["total_trials"] = total_trials
    result["heatmap_path"] = heatmap_path
    result["streaming_url"] = f"http://0.0.0.0:{STREAM_PORT}/"
    return result


@mcp.tool()
def quick_test_controller(controller_code: str) -> dict:
    """Sprint 4-5: コントローラを簡易テストする。
    グリッド評価なしで、中央位置のみで1回テスト（grid=1）。

    Args:
        controller_code: make_controller() 関数を含むPythonコード
    """
    from sandbox import validate_controller_code, execute_controller_safely

    valid, error = validate_controller_code(controller_code)
    if not valid:
        return {"success": False, "error": error, "survival_time": 0.0}

    result = execute_controller_safely(
        controller_code=controller_code,
        script_path=str(SCRIPTS_DIR / "04_survival_map.py"),
        script_args=["--no-stream", "--grid", "1"],
        timeout=30,
        cwd=str(WORKSHOP_DIR),
    )

    # Parse survival time from stdout
    # Pattern: "スコア: 10.0 秒" (with grid=1, score = single trial survival time)
    survival_time = 0.0
    for line in result.get("stdout", "").split("\n"):
        score_match = re.search(r'スコア[:\s]+([0-9.]+)\s*秒', line)
        if score_match:
            survival_time = float(score_match.group(1))

    result["survival_time"] = survival_time
    result["streaming_url"] = f"http://0.0.0.0:{STREAM_PORT}/"
    return result

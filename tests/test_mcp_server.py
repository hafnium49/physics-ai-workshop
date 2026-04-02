"""Tests for mujoco_mcp_server.py — MCP server logic.

No MuJoCo needed for most tests. Subprocess is mocked.
"""
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

# Add parent directory so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from mujoco_mcp_server import (
    SAFE_ENV,
    STREAM_PORT,
    MAX_SIM_DURATION,
    _wait_for_port,
    _wait_for_port_free,
    _stop_active_sim,
    _start_streaming_script,
    stop_simulation,
)


# ── TestSafeEnv ──────────────────────────────────────────────────────

class TestSafeEnv:
    """Test SAFE_ENV dictionary values."""

    def test_venv_bin_in_path(self):
        """SAFE_ENV PATH should contain .venv/bin."""
        assert ".venv/bin" in SAFE_ENV["PATH"]

    def test_mujoco_gl_is_osmesa(self):
        """MUJOCO_GL should default to osmesa."""
        # MUJOCO_GL comes from os.environ.get("MUJOCO_GL", "osmesa")
        # In test environment it should be osmesa unless overridden
        assert SAFE_ENV["MUJOCO_GL"] in ("osmesa", "egl", "glx"), (
            f"MUJOCO_GL should be a valid rendering backend, got: {SAFE_ENV['MUJOCO_GL']}"
        )

    def test_home_is_tmp(self):
        """HOME should be /tmp for security (prevent home directory access)."""
        assert SAFE_ENV["HOME"] == "/tmp"


# ── TestWaitForPort ──────────────────────────────────────────────────

class TestWaitForPort:
    """Test _wait_for_port() helper."""

    @patch("mujoco_mcp_server.socket.create_connection")
    def test_port_ready_immediately(self, mock_conn):
        """When socket connects immediately, return True."""
        mock_socket = MagicMock()
        mock_conn.return_value.__enter__ = MagicMock(return_value=mock_socket)
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)

        result = _wait_for_port(19999, timeout=1.0)
        assert result is True

    @patch("mujoco_mcp_server.socket.create_connection")
    @patch("mujoco_mcp_server.time.sleep")
    @patch("mujoco_mcp_server.time.monotonic")
    def test_port_timeout(self, mock_mono, mock_sleep, mock_conn):
        """When socket always refuses, return False after timeout."""
        # Simulate time: start at 0, advance past timeout on second call
        mock_mono.side_effect = [0.0, 0.0, 21.0]
        mock_conn.side_effect = ConnectionRefusedError()

        result = _wait_for_port(19999, timeout=20.0)
        assert result is False

    @patch("mujoco_mcp_server.socket.create_connection")
    def test_proc_dies_during_wait(self, mock_conn):
        """When proc.poll() returns non-None, return False immediately."""
        mock_conn.side_effect = ConnectionRefusedError()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # process exited with code 1

        result = _wait_for_port(19999, timeout=5.0, proc=mock_proc)
        assert result is False

    @patch("mujoco_mcp_server.socket.create_connection")
    @patch("mujoco_mcp_server.time.sleep")
    @patch("mujoco_mcp_server.time.monotonic")
    def test_proc_alive_port_opens(self, mock_mono, mock_sleep, mock_conn):
        """When proc is alive and port opens after retry, return True."""
        # Simulate: first try fails, second succeeds
        mock_socket = MagicMock()
        mock_conn.side_effect = [
            ConnectionRefusedError(),
            MagicMock(__enter__=MagicMock(return_value=mock_socket),
                      __exit__=MagicMock(return_value=False)),
        ]
        mock_mono.side_effect = [0.0, 0.0, 1.0, 2.0]
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # process is alive

        result = _wait_for_port(19999, timeout=20.0, proc=mock_proc)
        assert result is True


# ── TestWaitForPortFree ──────────────────────────────────────────────

class TestWaitForPortFree:
    """Test _wait_for_port_free() helper."""

    @patch("mujoco_mcp_server.socket.create_connection")
    def test_port_already_free(self, mock_conn):
        """ConnectionRefused immediately means port is free."""
        mock_conn.side_effect = ConnectionRefusedError()
        result = _wait_for_port_free(19999, timeout=1.0)
        assert result is True

    @patch("mujoco_mcp_server.socket.create_connection")
    @patch("mujoco_mcp_server.time.sleep")
    @patch("mujoco_mcp_server.time.monotonic")
    def test_port_becomes_free(self, mock_mono, mock_sleep, mock_conn):
        """Port is open first, then refuses -- should return True."""
        mock_socket = MagicMock()
        mock_conn.side_effect = [
            MagicMock(__enter__=MagicMock(return_value=mock_socket),
                      __exit__=MagicMock(return_value=False)),
            ConnectionRefusedError(),
        ]
        mock_mono.side_effect = [0.0, 0.0, 1.0, 2.0]

        result = _wait_for_port_free(19999, timeout=5.0)
        assert result is True

    @patch("mujoco_mcp_server.socket.create_connection")
    @patch("mujoco_mcp_server.time.sleep")
    @patch("mujoco_mcp_server.time.monotonic")
    def test_port_never_frees(self, mock_mono, mock_sleep, mock_conn):
        """Port always open -- should return False after timeout."""
        mock_socket = MagicMock()
        mock_conn.return_value = MagicMock(
            __enter__=MagicMock(return_value=mock_socket),
            __exit__=MagicMock(return_value=False),
        )
        # Exceed timeout on second iteration
        mock_mono.side_effect = [0.0, 0.0, 6.0]

        result = _wait_for_port_free(19999, timeout=5.0)
        assert result is False


# ── TestStopActiveSim ────────────────────────────────────────────────

class TestStopActiveSim:
    """Test _stop_active_sim() helper."""

    def test_stop_when_no_sim_running(self):
        """Should not crash when no simulation is active."""
        import mujoco_mcp_server
        # Reset global state
        with mujoco_mcp_server._sim_lock:
            mujoco_mcp_server._active_sim = None
            mujoco_mcp_server._auto_stop_timer = None
        # Should not raise
        _stop_active_sim()

    def test_stop_kills_process(self):
        """proc.terminate() should be called on the active sim."""
        import mujoco_mcp_server
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # process is alive
        mock_proc.wait.return_value = 0

        with mujoco_mcp_server._sim_lock:
            mujoco_mcp_server._active_sim = mock_proc
            mujoco_mcp_server._auto_stop_timer = None

        _stop_active_sim()
        mock_proc.terminate.assert_called_once()

    def test_stop_cancels_timer(self):
        """Timer.cancel() should be called if timer is active."""
        import mujoco_mcp_server
        mock_timer = MagicMock()

        with mujoco_mcp_server._sim_lock:
            mujoco_mcp_server._active_sim = None
            mujoco_mcp_server._auto_stop_timer = mock_timer

        _stop_active_sim()
        mock_timer.cancel.assert_called_once()

    def test_stop_reaps_zombie_after_kill(self):
        """proc.wait() should be called after kill() when terminate times out."""
        import mujoco_mcp_server
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        # terminate succeeds but wait times out, forcing kill
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="python", timeout=5),
            0,  # wait after kill succeeds
        ]

        with mujoco_mcp_server._sim_lock:
            mujoco_mcp_server._active_sim = mock_proc
            mujoco_mcp_server._auto_stop_timer = None

        _stop_active_sim()
        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()
        assert mock_proc.wait.call_count == 2


# ── TestStartStreamingScript ─────────────────────────────────────────

class TestStartStreamingScript:
    """Test _start_streaming_script() helper."""

    def _setup_mocks(self, mock_popen, mock_wait_port, mock_wait_free, mock_stop):
        """Common setup for start_streaming_script tests."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # process alive
        mock_popen.return_value = mock_proc
        mock_wait_port.return_value = True
        mock_wait_free.return_value = True
        return mock_proc

    @patch("mujoco_mcp_server._stop_active_sim")
    @patch("mujoco_mcp_server._wait_for_port_free")
    @patch("mujoco_mcp_server._wait_for_port")
    @patch("mujoco_mcp_server.subprocess.Popen")
    @patch("builtins.open", MagicMock())
    def test_starts_process_without_no_stream(self, mock_popen, mock_wait_port,
                                               mock_wait_free, mock_stop):
        """The command should NOT contain --no-stream."""
        self._setup_mocks(mock_popen, mock_wait_port, mock_wait_free, mock_stop)

        _start_streaming_script("01_validate_assembly.py")

        call_args = mock_popen.call_args
        cmd = call_args[0][0] if call_args[0] else call_args[1]["cmd"]
        assert "--no-stream" not in cmd

    @patch("mujoco_mcp_server._stop_active_sim")
    @patch("mujoco_mcp_server._wait_for_port_free")
    @patch("mujoco_mcp_server._wait_for_port")
    @patch("mujoco_mcp_server.subprocess.Popen")
    @patch("builtins.open", MagicMock())
    def test_sets_auto_timeout_timer(self, mock_popen, mock_wait_port,
                                      mock_wait_free, mock_stop):
        """Timer should be started with MAX_SIM_DURATION."""
        import mujoco_mcp_server
        self._setup_mocks(mock_popen, mock_wait_port, mock_wait_free, mock_stop)

        _start_streaming_script("01_validate_assembly.py")

        timer = mujoco_mcp_server._auto_stop_timer
        assert timer is not None
        # Clean up: cancel the timer to avoid it firing during tests
        timer.cancel()

    @patch("mujoco_mcp_server._stop_active_sim")
    @patch("mujoco_mcp_server._wait_for_port_free")
    @patch("mujoco_mcp_server._wait_for_port")
    @patch("mujoco_mcp_server.subprocess.Popen")
    @patch("builtins.open", MagicMock())
    def test_returns_streaming_url_on_success(self, mock_popen, mock_wait_port,
                                               mock_wait_free, mock_stop):
        """On success: success=True, streaming=True."""
        import mujoco_mcp_server
        self._setup_mocks(mock_popen, mock_wait_port, mock_wait_free, mock_stop)

        result = _start_streaming_script("01_validate_assembly.py")

        assert result["success"] is True
        assert result["streaming"] is True
        assert "streaming_url" in result
        # Clean up timer
        if mujoco_mcp_server._auto_stop_timer:
            mujoco_mcp_server._auto_stop_timer.cancel()

    @patch("mujoco_mcp_server._stop_active_sim")
    @patch("mujoco_mcp_server._wait_for_port_free")
    @patch("mujoco_mcp_server._wait_for_port")
    @patch("mujoco_mcp_server.subprocess.Popen")
    @patch("builtins.open", MagicMock())
    def test_returns_error_on_port_timeout(self, mock_popen, mock_wait_port,
                                            mock_wait_free, mock_stop):
        """When port never opens: success=False."""
        import mujoco_mcp_server
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # process alive but port never opens
        mock_popen.return_value = mock_proc
        mock_wait_port.return_value = False
        mock_wait_free.return_value = True

        result = _start_streaming_script("01_validate_assembly.py")

        assert result["success"] is False
        assert result["streaming"] is False
        # Clean up
        if mujoco_mcp_server._auto_stop_timer:
            mujoco_mcp_server._auto_stop_timer.cancel()

    @patch("mujoco_mcp_server._stop_active_sim")
    @patch("mujoco_mcp_server._wait_for_port_free")
    @patch("mujoco_mcp_server._wait_for_port")
    @patch("mujoco_mcp_server.subprocess.Popen")
    @patch("builtins.open", MagicMock())
    def test_returns_stderr_on_crash(self, mock_popen, mock_wait_port,
                                      mock_wait_free, mock_stop):
        """When process crashes: stderr from log included in error."""
        import mujoco_mcp_server
        mock_proc = MagicMock()
        # Process dies: poll returns exit code
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1
        mock_popen.return_value = mock_proc
        mock_wait_port.return_value = False
        mock_wait_free.return_value = True

        with mujoco_mcp_server._sim_lock:
            mujoco_mcp_server._active_sim = mock_proc

        result = _start_streaming_script("01_validate_assembly.py")

        assert result["success"] is False
        # Clean up
        if mujoco_mcp_server._auto_stop_timer:
            mujoco_mcp_server._auto_stop_timer.cancel()

    @patch("mujoco_mcp_server._stop_active_sim")
    @patch("mujoco_mcp_server._wait_for_port_free")
    @patch("mujoco_mcp_server._wait_for_port")
    @patch("mujoco_mcp_server.subprocess.Popen")
    @patch("builtins.open", MagicMock())
    def test_kills_previous_sim_before_starting(self, mock_popen, mock_wait_port,
                                                 mock_wait_free, mock_stop):
        """_stop_active_sim() should be called before starting new sim."""
        self._setup_mocks(mock_popen, mock_wait_port, mock_wait_free, mock_stop)

        import mujoco_mcp_server
        _start_streaming_script("01_validate_assembly.py")

        mock_stop.assert_called()
        # Clean up
        if mujoco_mcp_server._auto_stop_timer:
            mujoco_mcp_server._auto_stop_timer.cancel()

    @patch("mujoco_mcp_server._stop_active_sim")
    @patch("mujoco_mcp_server._wait_for_port_free")
    @patch("mujoco_mcp_server._wait_for_port")
    @patch("mujoco_mcp_server.subprocess.Popen")
    @patch("builtins.open", MagicMock())
    def test_waits_for_port_free_after_kill(self, mock_popen, mock_wait_port,
                                             mock_wait_free, mock_stop):
        """_wait_for_port_free() should be called after stopping previous sim."""
        self._setup_mocks(mock_popen, mock_wait_port, mock_wait_free, mock_stop)

        import mujoco_mcp_server
        _start_streaming_script("01_validate_assembly.py")

        mock_wait_free.assert_called()
        # Clean up
        if mujoco_mcp_server._auto_stop_timer:
            mujoco_mcp_server._auto_stop_timer.cancel()


# ── TestRelativeUrls ─────────────────────────────────────────────────

class TestRelativeUrls:
    """Test that the HTML page uses relative URLs for iframe compatibility."""

    def test_html_page_has_relative_stream_url(self):
        """Stream img src should be relative (no leading slash)."""
        from mujoco_streamer import _HTML_PAGE
        html = _HTML_PAGE.decode("utf-8") if isinstance(_HTML_PAGE, bytes) else _HTML_PAGE
        # Should use src="stream" not src="/stream"
        assert 'src="stream"' in html
        assert 'src="/stream"' not in html

    def test_html_page_has_relative_camera_url(self):
        """Camera fetch URL should be relative (no leading slash)."""
        from mujoco_streamer import _HTML_PAGE
        html = _HTML_PAGE.decode("utf-8") if isinstance(_HTML_PAGE, bytes) else _HTML_PAGE
        # Should use fetch('camera' not fetch('/camera'
        assert "fetch('camera'" in html or 'fetch("camera"' in html
        assert "fetch('/camera'" not in html and 'fetch("/camera"' not in html


# ── TestEvaluationTools ──────────────────────────────────────────────

class TestEvaluationTools:
    """Test evaluate_controller and quick_test_controller score parsing."""

    @patch("sandbox.execute_controller_safely")
    @patch("sandbox.validate_controller_code", return_value=(True, ""))
    def test_evaluate_controller_parses_score(self, mock_validate, mock_exec):
        """Regex should extract controller_score from stdout."""
        from mujoco_mcp_server import evaluate_controller

        mock_exec.return_value = {
            "success": True,
            "stdout": "処理中...\nスコア: 3.8 秒\n完全維持: 156/400 (39.0%)\n保存しました: survival_map.png",
            "stderr": "",
            "returncode": 0,
        }

        result = evaluate_controller("def make_controller(m,d,h):\n  pass\n")
        assert result["controller_score"] == 3.8

    @patch("sandbox.execute_controller_safely")
    @patch("sandbox.validate_controller_code", return_value=(True, ""))
    def test_evaluate_controller_parses_perfect_count(self, mock_validate, mock_exec):
        """Regex should extract perfect_count from stdout."""
        from mujoco_mcp_server import evaluate_controller

        mock_exec.return_value = {
            "success": True,
            "stdout": "スコア: 5.2 秒\n完全維持: 200/400 (50.0%)",
            "stderr": "",
            "returncode": 0,
        }

        result = evaluate_controller("def make_controller(m,d,h):\n  pass\n")
        assert result["perfect_count"] == 200
        assert result["total_trials"] == 400

    def test_evaluate_controller_empty_code_returns_zero(self):
        """Empty code should fail validation and return score 0."""
        from mujoco_mcp_server import evaluate_controller

        result = evaluate_controller("")
        assert result["success"] is False
        assert result["controller_score"] == 0.0

    @patch("sandbox.execute_controller_safely")
    @patch("sandbox.validate_controller_code", return_value=(True, ""))
    def test_quick_test_parses_survival_time(self, mock_validate, mock_exec):
        """Regex should extract survival_time from stdout."""
        from mujoco_mcp_server import quick_test_controller

        mock_exec.return_value = {
            "success": True,
            "stdout": "スコア: 10.0 秒\n",
            "stderr": "",
            "returncode": 0,
        }

        result = quick_test_controller("def make_controller(m,d,h):\n  pass\n")
        assert result["survival_time"] == 10.0

    def test_quick_test_empty_code_returns_zero(self):
        """Empty code should fail validation and return survival_time 0."""
        from mujoco_mcp_server import quick_test_controller

        result = quick_test_controller("")
        assert result["success"] is False
        assert result["survival_time"] == 0.0


# ── TestStopSimulation ───────────────────────────────────────────────

class TestStopSimulation:
    """Test the stop_simulation MCP tool."""

    def test_stop_when_running(self):
        """When a simulation is running, should return success with stop message."""
        import mujoco_mcp_server
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # process alive
        mock_proc.wait.return_value = 0

        with mujoco_mcp_server._sim_lock:
            mujoco_mcp_server._active_sim = mock_proc
            mujoco_mcp_server._auto_stop_timer = None

        result = stop_simulation()
        assert result["success"] is True
        assert "停止しました" in result["message"]

    def test_stop_when_not_running(self):
        """When no simulation running, should return message indicating that."""
        import mujoco_mcp_server
        with mujoco_mcp_server._sim_lock:
            mujoco_mcp_server._active_sim = None
            mujoco_mcp_server._auto_stop_timer = None

        result = stop_simulation()
        assert result["success"] is True
        assert "ありませんでした" in result["message"]

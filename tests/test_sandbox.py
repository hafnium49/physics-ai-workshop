"""Tests for sandbox.py — AST validation + subprocess isolation.

No MuJoCo needed. No MCP server needed. Pure Python.
"""
import re
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add parent directory so we can import sandbox
sys.path.insert(0, str(Path(__file__).parent.parent))

from sandbox import (
    validate_controller_code,
    execute_controller_safely,
    FORBIDDEN_IMPORTS,
    FORBIDDEN_CALLS,
    FORBIDDEN_ATTRS,
    MAX_CODE_SIZE,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _has_cjk(text: str) -> bool:
    """Return True if text contains CJK (Japanese/Chinese/Korean) characters."""
    return bool(re.search(r'[\u3000-\u9fff\uf900-\ufaff]', text))


# ── TestValidateControllerCode ───────────────────────────────────────

class TestValidateControllerCode:
    """Tests for validate_controller_code()."""

    # -- Happy path --

    def test_valid_code_with_make_controller(self, valid_controller_code):
        ok, err = validate_controller_code(valid_controller_code)
        assert ok is True
        assert err == ""

    def test_valid_code_with_numpy_import(self, valid_code_with_numpy):
        ok, err = validate_controller_code(valid_code_with_numpy)
        assert ok is True
        assert err == ""

    def test_valid_code_with_closure(self):
        code = (
            "def make_controller(model, dt, home):\n"
            "    gains = [1.0, 2.0]\n"
            "    def helper(x):\n"
            "        return x * gains[0]\n"
            "    def controller(data, plate_id, ball_id, step, t):\n"
            "        v = helper(data.ctrl[0])\n"
            "        data.ctrl[0] = v\n"
            "    return controller\n"
        )
        ok, err = validate_controller_code(code)
        assert ok is True
        assert err == ""

    # -- Code size --

    def test_code_exceeds_max_size(self, valid_controller_code):
        oversized = valid_controller_code + " " * (MAX_CODE_SIZE + 1)
        ok, err = validate_controller_code(oversized)
        assert ok is False
        assert _has_cjk(err)

    def test_code_at_max_size(self):
        base = (
            "def make_controller(model, dt, home):\n"
            "    def controller(data, plate_id, ball_id, step, t):\n"
            "        pass\n"
            "    return controller\n"
        )
        # Pad with comments to reach exactly MAX_CODE_SIZE
        padding = "# " + "x" * (MAX_CODE_SIZE - len(base) - 3) + "\n"
        code = padding + base
        assert len(code) == MAX_CODE_SIZE
        ok, err = validate_controller_code(code)
        assert ok is True
        assert err == ""

    # -- Syntax errors --

    def test_syntax_error_returns_false(self):
        ok, err = validate_controller_code("def make_controller(model, dt, home\n")
        assert ok is False
        assert "構文エラー" in err

    def test_empty_string(self):
        ok, err = validate_controller_code("")
        assert ok is False
        # No make_controller found
        assert _has_cjk(err)

    # -- Missing make_controller --

    def test_no_make_controller_function(self):
        code = "def some_other_function():\n    pass\n"
        ok, err = validate_controller_code(code)
        assert ok is False
        assert "make_controller" in err

    # -- Forbidden imports (individual checks) --

    def test_forbidden_import_os(self):
        code = "import os\ndef make_controller(model, dt, home):\n    pass\n"
        ok, err = validate_controller_code(code)
        assert ok is False
        assert "禁止" in err

    def test_forbidden_import_subprocess(self):
        code = "import subprocess\ndef make_controller(model, dt, home):\n    pass\n"
        ok, err = validate_controller_code(code)
        assert ok is False
        assert "禁止" in err

    def test_forbidden_from_import(self):
        code = "from os import path\ndef make_controller(model, dt, home):\n    pass\n"
        ok, err = validate_controller_code(code)
        assert ok is False
        assert "禁止" in err

    def test_allowed_import_numpy(self):
        code = "import numpy\ndef make_controller(model, dt, home):\n    pass\n"
        ok, err = validate_controller_code(code)
        assert ok is True

    def test_allowed_import_math(self):
        code = "import math\ndef make_controller(model, dt, home):\n    pass\n"
        ok, err = validate_controller_code(code)
        assert ok is True

    # -- Forbidden calls (individual checks) --

    def test_forbidden_call_exec(self):
        code = 'exec("pass")\ndef make_controller(model, dt, home):\n    pass\n'
        ok, err = validate_controller_code(code)
        assert ok is False
        assert "禁止" in err

    def test_forbidden_call_eval(self):
        code = 'eval("1+1")\ndef make_controller(model, dt, home):\n    pass\n'
        ok, err = validate_controller_code(code)
        assert ok is False
        assert "禁止" in err

    def test_forbidden_call_open(self):
        code = 'open("/etc/passwd")\ndef make_controller(model, dt, home):\n    pass\n'
        ok, err = validate_controller_code(code)
        assert ok is False
        assert "禁止" in err

    def test_forbidden_call___import__(self):
        code = '__import__("os")\ndef make_controller(model, dt, home):\n    pass\n'
        ok, err = validate_controller_code(code)
        assert ok is False
        assert "禁止" in err

    # -- Forbidden attributes (individual checks) --

    def test_forbidden_attr___subclasses__(self):
        code = (
            "def make_controller(model, dt, home):\n"
            "    x = object.__subclasses__\n"
            "    pass\n"
        )
        ok, err = validate_controller_code(code)
        assert ok is False
        assert "禁止" in err

    def test_forbidden_attr___globals__(self):
        code = (
            "def make_controller(model, dt, home):\n"
            "    x = f.__globals__\n"
            "    pass\n"
        )
        ok, err = validate_controller_code(code)
        assert ok is False
        assert "禁止" in err

    def test_forbidden_attr_system(self):
        code = (
            "def make_controller(model, dt, home):\n"
            "    x = os.system\n"
            "    pass\n"
        )
        ok, err = validate_controller_code(code)
        assert ok is False
        assert "禁止" in err

    # -- Japanese error messages --

    def test_error_messages_are_japanese(self):
        """All error paths should produce messages containing CJK characters."""
        # Oversized code
        _, err = validate_controller_code("x" * (MAX_CODE_SIZE + 1))
        assert _has_cjk(err), f"Oversized error not Japanese: {err}"

        # Syntax error
        _, err = validate_controller_code("def (:")
        assert _has_cjk(err), f"Syntax error not Japanese: {err}"

        # Missing make_controller
        _, err = validate_controller_code("x = 1\n")
        assert _has_cjk(err), f"Missing function error not Japanese: {err}"

        # Forbidden import
        _, err = validate_controller_code(
            "import os\ndef make_controller(model, dt, home):\n    pass\n"
        )
        assert _has_cjk(err), f"Forbidden import error not Japanese: {err}"

        # Forbidden call
        _, err = validate_controller_code(
            'exec("x")\ndef make_controller(model, dt, home):\n    pass\n'
        )
        assert _has_cjk(err), f"Forbidden call error not Japanese: {err}"

        # Forbidden attribute
        _, err = validate_controller_code(
            "def make_controller(model, dt, home):\n"
            "    x = o.__globals__\n"
        )
        assert _has_cjk(err), f"Forbidden attr error not Japanese: {err}"

    # -- Parametrized forbidden sets (covers ALL entries) --

    @pytest.mark.parametrize("mod", sorted(FORBIDDEN_IMPORTS))
    def test_forbidden_import_parametrized(self, mod):
        code = f"import {mod}\ndef make_controller(model, dt, home):\n    pass\n"
        ok, err = validate_controller_code(code)
        assert ok is False, f"import {mod} should be forbidden"
        assert "禁止" in err

    @pytest.mark.parametrize("func", sorted(FORBIDDEN_CALLS))
    def test_forbidden_call_parametrized(self, func):
        code = f'{func}("x")\ndef make_controller(model, dt, home):\n    pass\n'
        ok, err = validate_controller_code(code)
        assert ok is False, f"{func}() should be forbidden"
        assert "禁止" in err

    @pytest.mark.parametrize("attr", sorted(FORBIDDEN_ATTRS))
    def test_forbidden_attr_parametrized(self, attr):
        code = (
            f"def make_controller(model, dt, home):\n"
            f"    x = obj.{attr}\n"
            f"    pass\n"
        )
        ok, err = validate_controller_code(code)
        assert ok is False, f".{attr} should be forbidden"
        assert "禁止" in err


# ── TestExecuteControllerSafely ──────────────────────────────────────

class TestExecuteControllerSafely:
    """Tests for execute_controller_safely()."""

    def test_validation_failure_returns_early(self):
        """Invalid code should return success=False, returncode=-1 without running subprocess."""
        result = execute_controller_safely(
            controller_code="import os\ndef make_controller(m, d, h):\n    pass\n",
            script_path="/nonexistent/script.py",
        )
        assert result["success"] is False
        assert result["returncode"] == -1
        assert result["stderr"] != ""

    @patch("sandbox.subprocess.run")
    def test_timeout_returns_minus_2(self, mock_run, valid_controller_code):
        """subprocess.TimeoutExpired should return returncode=-2."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="python", timeout=120)

        result = execute_controller_safely(
            controller_code=valid_controller_code,
            script_path="/tmp/fake_script.py",
            timeout=120,
        )
        assert result["success"] is False
        assert result["returncode"] == -2
        assert "タイムアウト" in result["stderr"]

    @patch("sandbox.subprocess.run")
    def test_generic_exception_returns_minus_3(self, mock_run, valid_controller_code):
        """Generic Exception should return returncode=-3."""
        mock_run.side_effect = RuntimeError("unexpected failure")

        result = execute_controller_safely(
            controller_code=valid_controller_code,
            script_path="/tmp/fake_script.py",
        )
        assert result["success"] is False
        assert result["returncode"] == -3

    def test_temp_file_cleaned_up(self, valid_controller_code, tmp_path):
        """Controller temp file should be deleted in finally block."""
        # Use a non-existent script so subprocess.run will raise an error
        # but the temp file should still be cleaned up.
        with patch("sandbox.subprocess.run", side_effect=OSError("no such script")):
            execute_controller_safely(
                controller_code=valid_controller_code,
                script_path="/tmp/fake_script.py",
                cwd=str(tmp_path),
            )

        # After execution, no .py temp files should remain in tmp_path
        remaining_py = list(tmp_path.glob("*.py"))
        assert len(remaining_py) == 0, f"Temp file not cleaned up: {remaining_py}"

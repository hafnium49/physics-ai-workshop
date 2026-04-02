"""Controller code sandbox — AST validation + subprocess isolation."""

import ast
import os
import subprocess
import tempfile
from pathlib import Path

FORBIDDEN_IMPORTS = {
    'os', 'sys', 'subprocess', 'shutil', 'socket', 'http',
    'urllib', 'requests', 'pathlib', 'io', 'ctypes', 'signal',
    'multiprocessing', 'threading', 'pickle', 'marshal', 'code',
    'importlib', 'builtins',
}

FORBIDDEN_CALLS = {
    'exec', 'eval', 'compile', '__import__', 'open',
    'input', 'breakpoint', 'exit', 'quit',
}

FORBIDDEN_ATTRS = {
    '__subclasses__', '__bases__', '__globals__', '__code__',
    '__builtins__', '__import__', 'system', 'popen',
}

MAX_CODE_SIZE = 10 * 1024  # 10KB


def validate_controller_code(code: str) -> tuple[bool, str]:
    """Validate controller code via AST inspection.

    Returns (True, "") if safe, (False, "error message") if not.
    """
    if len(code) > MAX_CODE_SIZE:
        return False, f"コードサイズが上限を超えています（{len(code)} > {MAX_CODE_SIZE} bytes）"

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"構文エラー: {e}"

    # Check for make_controller definition
    has_make_controller = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "make_controller"
        for node in ast.walk(tree)
    )
    if not has_make_controller:
        return False, "make_controller() 関数が定義されていません"

    # Walk AST for forbidden patterns
    for node in ast.walk(tree):
        # Forbidden imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.split('.')[0]
                if mod in FORBIDDEN_IMPORTS:
                    return False, f"禁止されたインポート: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod = node.module.split('.')[0]
                if mod in FORBIDDEN_IMPORTS:
                    return False, f"禁止されたインポート: {node.module}"

        # Forbidden function calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                return False, f"禁止された関数呼び出し: {node.func.id}()"
            elif isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_CALLS:
                return False, f"禁止された関数呼び出し: .{node.func.attr}()"

        # Forbidden attribute access
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS:
                return False, f"禁止された属性アクセス: .{node.attr}"

    return True, ""


SAFE_ENV = {
    "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin"),
    "HOME": "/tmp",
    "MUJOCO_GL": os.environ.get("MUJOCO_GL", "osmesa"),
    "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
    "STREAM_PORT": os.environ.get("STREAM_PORT", "18080"),
}


def execute_controller_safely(
    controller_code: str,
    script_path: str,
    script_args: list[str] | None = None,
    timeout: int = 120,
    cwd: str | None = None,
) -> dict:
    """Execute a controller in a subprocess with stripped environment.

    Args:
        controller_code: Python source code with make_controller()
        script_path: Path to the evaluation script (e.g., 04_survival_map.py)
        script_args: Additional CLI arguments
        timeout: Process timeout in seconds
        cwd: Working directory

    Returns:
        {"success": bool, "stdout": str, "stderr": str, "returncode": int}
    """
    # Validate first
    valid, error = validate_controller_code(controller_code)
    if not valid:
        return {"success": False, "stdout": "", "stderr": error, "returncode": -1}

    # Write controller to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=cwd or "/tmp"
    ) as f:
        f.write(controller_code)
        controller_path = f.name

    try:
        cmd = ["python", script_path]
        if script_args:
            cmd.extend(script_args)
        cmd.extend(["--controller", controller_path])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=SAFE_ENV,
            cwd=cwd or str(Path(script_path).parent.parent),
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"タイムアウト: {timeout}秒を超えました",
            "returncode": -2,
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -3,
        }
    finally:
        Path(controller_path).unlink(missing_ok=True)

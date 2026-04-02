"""FastMCP server for Physics-AI Workshop — wraps MuJoCo simulation scripts."""

import json
import os
import re
import subprocess
from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP("physics-workshop")

WORKSHOP_DIR = Path(__file__).parent
SCRIPTS_DIR = WORKSHOP_DIR / "scripts"
STREAM_PORT = int(os.environ.get("STREAM_PORT", "18080"))
MUJOCO_GL = os.environ.get("MUJOCO_GL", "egl")

# Safe environment for subprocess execution
SAFE_ENV = {
    "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin"),
    "HOME": "/tmp",
    "MUJOCO_GL": MUJOCO_GL,
    "PYTHONPATH": str(WORKSHOP_DIR),
    "STREAM_PORT": str(STREAM_PORT),
}


def _run_script(script_name: str, args: list[str] | None = None, timeout: int = 120) -> dict:
    """Run a workshop script and capture output."""
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


@mcp.tool()
def validate_assembly(duration: float = 3.0) -> dict:
    """Sprint 1: ロボットアームとプレートの構成を確認する。
    シミュレーションを実行し、ライブストリームURLを返す。

    Args:
        duration: シミュレーション時間（秒）。デフォルト: 3.0
    """
    result = _run_script("01_validate_assembly.py", [
        "--no-stream",
        "--duration", str(duration),
    ], timeout=30)

    result["streaming_url"] = f"http://0.0.0.0:{STREAM_PORT}/"
    return result


@mcp.tool()
def run_pid_controller(
    kp: float = 50.0,
    kd: float = 10.0,
    script: str = "02_pid_baseline.py",
) -> dict:
    """Sprint 2-3: PIDコントローラでボールバランスを実行する。

    Args:
        kp: 比例ゲイン。デフォルト: 50.0（02_pid_baseline用）。03_optimize_pidでは無視される。
        kd: 微分ゲイン。デフォルト: 10.0（02_pid_baseline用）。03_optimize_pidでは無視される。
        script: 使用するスクリプト。"02_pid_baseline.py" (意図的に壊れたPID) or "03_optimize_pid.py" (正しいPID)
    """
    args = ["--no-stream"]
    if script == "02_pid_baseline.py":
        args.extend(["--kp", str(kp), "--kd", str(kd)])
    result = _run_script(script, args, timeout=60)

    # Parse survival times from stdout
    # Pattern: "維持時間: 3.5 秒" or "維持時間: 10.0 秒"
    survival_times = []
    for line in result.get("stdout", "").split("\n"):
        match = re.search(r'維持時間[:\s]+([0-9.]+)\s*秒', line)
        if match:
            survival_times.append(float(match.group(1)))

    result["survival_times"] = survival_times
    result["mean_survival"] = sum(survival_times) / len(survival_times) if survival_times else 0.0
    result["streaming_url"] = f"http://0.0.0.0:{STREAM_PORT}/"
    return result


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

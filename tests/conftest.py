"""Shared fixtures for physics-ai-workshop tests."""
import pytest


@pytest.fixture
def valid_controller_code():
    """Minimal valid controller code matching the workshop interface."""
    return (
        "def make_controller(model, dt, home):\n"
        "    def controller(data, plate_id, ball_id, step, t):\n"
        "        pass\n"
        "    return controller\n"
    )


@pytest.fixture
def valid_code_with_numpy():
    """Valid controller code that imports numpy (allowed)."""
    return (
        "import numpy as np\n"
        "def make_controller(model, dt, home):\n"
        "    Kp = 2.0\n"
        "    def controller(data, plate_id, ball_id, step, t):\n"
        "        bx = data.xpos[ball_id][0] - data.xpos[plate_id][0]\n"
        "        data.ctrl[5] = home[5] + Kp * bx\n"
        "    return controller\n"
    )

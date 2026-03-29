# FY2026 Physics-AI Workshop

Building Digital Twins with AI Coding Agents

## Overview

A 1-hour hands-on workshop where material engineers use [Claude Code](https://claude.ai/code) (an AI coding agent) to build and optimize MuJoCo physics simulations on an NVIDIA DGX Spark.

No programming experience required. You tell the AI what to build in plain English, and it writes the code for you.

## What You'll Build

A **Franka Panda robotic arm** holding a plate with a ball on top. Your goal: use Claude Code to assemble the simulation, test the physics, and optimize PID control parameters until the ball stays balanced for 10 seconds.

## Workshop Structure

| Sprint | Time | Goal | What You'll Do |
|--------|------|------|----------------|
| 1. Explore | 15 min | Understand the Robot | Run the pre-built simulation, move joints, build intuition |
| 2. PID Discovery | 12 min | Teach the Robot to Balance | Run a broken controller, ask Claude to diagnose and fix it |
| 3. Challenges | 8 min | Test Robustness | Add disturbances (pushes, oscillation) and see if the ball survives |
| 4. Free Exploration | 25 min | Improve the Controller | Use the survival map as your scoreboard. Ask Claude to try better control strategies |

## Prerequisites

- **Visual Studio Code** installed on your laptop
- **Remote - SSH** extension installed in VS Code
- Connection details will be provided on a printed card at the workshop

## Repository Structure

```
physics-ai-workshop/
├── content/                # Simulation models
│   ├── ball_and_plate.xml  # Plate + ball (your balancing target)
│   └── franka_panda/       # Franka Panda robotic arm (from MuJoCo Menagerie)
├── docs/
│   ├── participant-guide.md    # Step-by-step workshop guide
│   └── host-preparation-runbook.md  # For the workshop host/admin
└── CLAUDE.md               # Context for Claude Code sessions
```

## Quick Start

After connecting via VS Code (see your printed card):

1. Open a terminal: `Ctrl + ~`
2. Verify your prompt shows `(workshop_env)`
3. Navigate to your workspace:
   ```bash
   cd ~/physics_sim
   ```
4. Start Claude Code:
   ```bash
   claude
   ```
5. Give your first instruction:
   > "Run the 01_validate_assembly.py script and start the live stream so I can see the robot."

## Simulation Content

### Franka Panda Arm

A 7-DOF robotic arm from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — a collection of high-quality open-source robot models maintained by Google DeepMind. The model includes accurate meshes, joint limits, and dynamics parameters.

### Ball and Plate

A minimal model defining a flat plate and a ball with a free joint. The workshop task is to attach this to the Panda's gripper and keep the ball balanced through PID control.

## Resources

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [Claude Code](https://claude.ai/code)
- [MuJoCo Python Bindings](https://mujoco.readthedocs.io/en/stable/python.html)

## License

This workshop repository is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0). The Franka Panda model is from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Apache 2.0, Copyright Google DeepMind).

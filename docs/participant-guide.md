# Participant Guide — FY2026 Physics-AI Workshop

Welcome! In this 1-hour workshop, you'll use an AI coding agent to build a physics simulation of a robotic arm balancing a ball on a plate.

**No programming experience needed.** You describe what you want in plain English, and the AI writes the code.

---

## Before the Workshop

Install these on your corporate laptop:

1. **Visual Studio Code** — [Download here](https://code.visualstudio.com/)
2. **Remote - SSH extension** — Open VS Code, go to Extensions (`Ctrl+Shift+X`), search "Remote - SSH", install it

That's it. Everything else is pre-configured on the server.

---

## Connecting (Workshop Day)

You'll receive a **printed card** with your personal connection details and password.

1. Open **VS Code**
2. Click the green **`><`** icon in the bottom-left corner
3. Select **"Connect to Host..."**
4. Type the connection command from your card and press **Enter**
5. When prompted, enter the password from your card
6. Wait ~20 seconds for VS Code to finish setting up

You should see the bottom-left of VS Code change to show you're connected remotely.

---

## Getting Started

1. **Open a terminal:** Press `Ctrl + ~` (or go to Terminal → New Terminal)
2. **Check your environment:** Your prompt should show `(workshop_env)` — this means your Python environment is ready
3. **Go to your workspace:**
   ```
   cd ~/physics_sim
   ```
4. **Start the AI agent:**
   ```
   claude
   ```

You're now talking to Claude Code. It can read files, write code, and run simulations for you.

---

## Your First Task

Copy and paste this into Claude:

> Read the Panda robot XML file in content/franka_panda/panda.xml and the content/ball_and_plate.xml file. Combine them so the plate is attached to the robot's end-effector, with the ball sitting on top. Run a short simulation and save the result as simulation.mp4

Claude will write a Python script, run it, and produce a video file you can view.

---

## Sprint Guide

### Sprint 1: Assembly (15 minutes)

**Goal:** Build your simulation world.

Tell Claude to:
- Read and combine the robot arm with the plate and ball
- Run a simulation to verify everything is connected
- Save the output as a video

### Sprint 2: Baseline (15 minutes)

**Goal:** Test the physics.

Tell Claude to:
- Drop the ball onto the plate from a small height
- Add a simple PID controller to try to keep the ball balanced
- Record what happens as a video

You'll likely see the ball fall off — that's expected!

### Sprint 3: Optimization (30 minutes)

**Goal:** Make it work.

Tell Claude to:
- Systematically tune the Kp (proportional) and Kd (derivative) parameters
- Run multiple experiments and track the results
- Find parameters that keep the ball centered for 10 seconds
- Save the best result as a video

Feel free to ask Claude questions like:
- *"What is PID control?"*
- *"Why does the ball keep falling to the left?"*
- *"Can you try 10 different Kp values and show me which one works best?"*

---

## Tips

- **Always ask for video output.** Since we're working over SSH, there are no graphical windows. All results should be saved as `.mp4` files.
- **Ask questions freely.** Claude can explain any concept — MuJoCo, PID control, physics parameters, Python code.
- **If something breaks,** just tell Claude: *"That didn't work. Here's the error:"* and paste the error message.
- **Be specific about what you want.** Instead of "make it better," try "increase Kp to 100 and decrease Kd to 5."

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| VS Code disconnected | Click the green `><` icon → "Connect to Host" → reconnect |
| Terminal is frozen | Open a new terminal: Terminal → New Terminal |
| Claude is not responding | Press `Ctrl+C` to cancel, then type `claude` to restart |
| Can't find the video file | Ask Claude: "Where did you save the video?" |
| `(workshop_env)` not showing | Run: `source ~/workshop_env/bin/activate` |

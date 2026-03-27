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

> Load the panda_ball_balance.xml model. Write a Python script that runs the simulation with the arm holding the plate and ball. Use the mujoco_streamer.py helper to stream the video live on port 8080. Hold the arm at the home pose and let me watch the ball.

Claude will write a Python script, run it, and start a live video stream you can watch in your browser.

---

## Sprint Guide

### Sprint 1: Explore the Model (15 minutes)

**Goal:** Get familiar with the robot and its joints.

The simulation model is already pre-built for you — no assembly required.

Tell Claude to:
- Load `panda_ball_balance.xml` and start a live stream so you can watch
- Move individual joints one at a time so you can see which ones control the plate tilt
- Explain what each joint does as it moves

### Sprint 2: PID Discovery (15 minutes)

**Goal:** Find out which joints actually matter for balancing.

Tell Claude to:
- Write a PID controller for the wrist joints to keep the ball centered
- Run the simulation and watch it live

It will probably fail — the ball will roll off. That's the point! Now ask Claude:
- *"Which joints actually tilt the plate?"*
- *"Can you try controlling different joints?"*

Keep iterating with Claude until you find the right joints and the right direction (sign) for the corrections.

### Sprint 3: Progressive Challenges (30 minutes)

**Goal:** Make it robust.

**Challenge 1 — Survive a push.** Tell Claude to:
- Add periodic force disturbances that push the ball sideways
- Tune the Kp and Kd gains until the ball stays on the plate for 10 seconds despite the pushes

**Challenge 2 — Stronger pushes.** Tell Claude to:
- Increase the disturbance strength
- Adjust the gains to compensate

**Challenge 3 (advanced) — Oscillating plate.** Tell Claude to:
- Add a slow oscillation to the plate while also applying disturbances
- Find gains that keep the ball centered through both

Feel free to ask Claude questions like:
- *"What is PID control?"*
- *"Why does the ball keep falling to the left?"*
- *"Can you try 10 different Kp values and show me which one works best?"*

---

## Tips

- **Ask Claude to use the `mujoco_streamer.py` helper for live viewing in your browser.** Since we're working over SSH, there are no graphical windows — the streamer sends video to your browser instead.
- **If the video stream stops,** just tell Claude to restart it.
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
| Can't see the live video | Click the port forwarding popup in VS Code, or go to the Ports tab and click the globe icon |
| `(workshop_env)` not showing | Run: `source ~/workshop_env/bin/activate` |

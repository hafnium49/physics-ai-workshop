# Participant Guide — FY2026 Physics-AI Workshop

Welcome! In this 1-hour workshop, you'll use an AI coding agent to build a physics simulation of a robotic arm balancing a ball on a plate — and watch it live in your browser.

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

You're now talking to Claude Code. It already knows about the workshop, the robot model, and the streaming setup — just tell it what you want to do.

> **Your streaming port** is shown on your printed card (e.g., 18081). When Claude asks about ports or you want to specify one, use this number.

---

## What You Just Watched

Before the workshop started, the host ran a live demo where Claude Code discovered — by itself — which joints control the plate tilt. It tried the wrong joints, diagnosed the problem, switched to the right ones, and balanced the ball for 10 seconds. **Now it's your turn.**

---

## How Live Viewing Works

Since we're working over SSH, there are no graphical windows. Instead, your simulation runs on the server and **streams live video to your browser**.

When Claude runs a simulation script, you'll see a pop-up in the bottom-right corner of VS Code saying something like *"Your application running on port 18081 is available."* Click **"Open in Browser"** to watch the robot simulation in real time.

The browser view is **interactive** — you can drag to rotate the view, scroll to zoom in/out, and right-drag to pan. Press **R** to reset the camera to the default angle. This works while the simulation is running.

If you miss the pop-up, go to the **Ports** tab at the bottom of VS Code and click the globe icon next to the port number.

---

## Sprint 1: Explore the Robot (15 minutes)

**Goal:** See the robot and understand which joints control the plate.

The simulation model is already pre-built — a Franka Panda robotic arm holding a plate with a ball on top. No assembly required.

### Step 1 — See the robot

Tell Claude:

> Run scripts/01_validate_assembly.py and start the live stream.

Watch the browser — you'll see the robot arm, the plate, and the ball rolling off. That's expected! The arm isn't actively balancing yet.

### Step 2 — Explore the joints

Now tell Claude:

> Stop the simulation. Move joint 6 slowly, then joint 7, then joint 5. Which ones tilt the plate?

Watch each joint move. Notice which ones tilt the plate and which ones barely do anything. **This matters for the next sprint.**

---

## Sprint 2: PID Discovery (12 minutes)

**Goal:** Discover why the baseline controller fails and fix it.

### Step 1 — Run the broken controller

Tell Claude:

> Run scripts/02_pid_baseline.py and start the live stream.

Watch the live stream. The ball falls off in less than 1 second. Look at the **Survival Time** printed in the terminal.

### Step 2 — Diagnose and fix

Tell Claude:

> The ball keeps falling off immediately. Can you check which joints actually control the plate tilt? And check if the correction sign is right.

Claude will analyze the joints and the correction direction. Once it finds the right joints and fixes the sign, survival time should jump to 10 seconds.

### Step 3 — Understand what changed

Tell Claude:

> Explain what you changed and why.

This is the key insight: **which joints to control and the sign of the correction matter more than the gain values.**

---

## Sprint 3: Challenges (8 minutes)

**Goal:** Test the controller under disturbances.

Now that the ball stays balanced on a still plate, let's make it harder.

Tell Claude:

> Run scripts/04_challenge.py --level 2 and start the live stream.

Watch the ball get pushed around by random forces. Does it survive?

**Optional:** Try Level 3 (oscillation):

> Run scripts/04_challenge.py --level 3 and start the live stream.

---

## Sprint 4: Free Exploration (25 minutes)

**Goal:** Improve the controller beyond basic PID.

### Get your baseline

Tell Claude:

> Run scripts/05_survival_map.py and show me the survival map.

The survival map shows where the ball can start on the plate and survive 10 seconds (green) vs. where it falls off (red). Your goal: **make the green zone cover more of the plate.**

### Try improvements

Here are some things you can ask Claude in plain English:

- *"Can you try different PID gain values and compare survival maps?"*
- *"The ball keeps falling off when it starts near the edge. Can you make the controller react faster when the ball is far from center?"*
- *"Is there a better control method than PID? Can you try one and compare?"*
- *"Can the controller predict where the ball is going instead of just reacting?"*
- *"What if the gains were different for X vs Y directions?"*

### Competition

How many grid positions survive 10 seconds? Check with other participants!

Feel free to ask Claude anything along the way:
- *"What is PID control?"*
- *"Why does the ball keep falling to the left?"*
- *"What happens if I set Kd to zero?"*
- *"Can you try 10 different Kp values and show me which is best?"*

---

## Tips

- **Watch the browser, not the terminal.** The live video stream is where you see the robot. The terminal shows numbers and status.
- **If the video stream stops,** just tell Claude: *"Restart the live stream."*
- **Ask questions freely.** Claude can explain physics, control theory, or anything about the simulation.
- **If something breaks,** tell Claude: *"That didn't work. Here's the error:"* and paste the error message.
- **Be specific.** Instead of *"make it better,"* try *"increase Kp to 100 and decrease Kd to 5."*
- **Ctrl+C stops the simulation.** If you want to try something new, press `Ctrl+C` in the terminal to stop the current script.
- **If you see "Port already in use",** it means a previous script is still running. Press `Ctrl+C` in that terminal first.
- **Rotate the camera** by left-dragging in the browser. Scroll to zoom, right-drag to pan. Press R to reset the view.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| VS Code disconnected | Click the green `><` icon → "Connect to Host" → reconnect |
| Terminal is frozen | Open a new terminal: Terminal → New Terminal |
| Claude is not responding | Press `Ctrl+C` to cancel, then type `claude` to restart |
| Can't see the live video | Click the port forwarding popup in VS Code, or go to the Ports tab and click the globe icon |
| Video shows but is frozen | Tell Claude: "Restart the live stream" |
| "Port already in use" error | Another script is still running. Press `Ctrl+C` in the other terminal first, then retry |
| `(workshop_env)` not showing | Run: `source ~/workshop_env/bin/activate` |

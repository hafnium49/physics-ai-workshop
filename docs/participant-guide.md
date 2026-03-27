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

You're now talking to Claude Code. It can read files, write code, and run simulations for you.

---

## How Live Viewing Works

Since we're working over SSH, there are no graphical windows. Instead, your simulation runs on the server and **streams live video to your browser**.

When Claude runs a simulation script, you'll see a pop-up in the bottom-right corner of VS Code saying something like *"Your application running on port 8080 is available."* Click **"Open in Browser"** to watch the robot simulation in real time.

If you miss the pop-up, go to the **Ports** tab at the bottom of VS Code and click the globe icon next to the port number.

---

## Sprint 1: Explore the Robot (15 minutes)

**Goal:** See the robot and understand which joints control the plate.

The simulation model is already pre-built — a Franka Panda robotic arm holding a plate with a ball on top. No assembly required.

### Step 1 — See the robot

Copy and paste this into Claude:

> Read the panda_ball_balance.xml model file. Write a Python script that loads this model, places the ball on the plate, and streams the simulation live using the mujoco_streamer.py helper. Hold the arm at the home pose. Let me watch the ball fall off naturally.

Watch the browser — you'll see the robot arm, the plate, and the ball rolling off. That's expected! The arm isn't actively balancing yet.

### Step 2 — Explore the joints

Now tell Claude:

> Stop the simulation. I want to understand the robot's joints. Write a new script that moves joint 5 slowly back and forth by 0.1 radians while streaming live. Then do the same for joint 6 and joint 7, one at a time. Tell me what each joint does to the plate.

Watch each joint move. Notice which ones tilt the plate and which ones barely do anything. **This matters for the next sprint.**

---

## Sprint 2: Teach the Robot to Balance (15 minutes)

**Goal:** Build a controller that keeps the ball on the plate.

### Step 1 — First attempt

Tell Claude:

> Write a PID controller that tries to keep the ball centered on the plate. Use the ball's position relative to the plate center as the error signal. Pick whichever wrist joints you think control the plate tilt. Stream the simulation live and print "Survival Time: X.X seconds" when the ball falls off. Auto-reset and try again.

Watch the live stream. The ball will probably fall off quickly. Look at the **Survival Time** printed in the terminal.

### Step 2 — Diagnose and fix

If the ball falls off fast (under 2 seconds), ask Claude:

> The ball fell off in less than 2 seconds. Can you check which joints actually move the plate? Try nudging each wrist joint by a small amount and measure how much the plate position changes. Then switch to the joints that have the most effect.

Claude will test each joint's authority over the plate and discover the right ones. Once it switches to the correct joints and sign, survival time should jump dramatically.

### Step 3 — Confirm it works

Once the ball stays on for 10 seconds:

> Great! The ball is balancing. Can you explain which joints you're controlling and why? What did you change from the first attempt?

This is the key insight: **which joints to control matters more than the gain values.**

---

## Sprint 3: Progressive Challenges (30 minutes)

**Goal:** Make the balancing robust under disturbances.

Now that the ball stays balanced on a still plate, let's make it harder.

### Challenge 1 — Survive a push

> Add random force disturbances that push the ball sideways every 2 seconds. Start with a force magnitude of 0.5 Newtons. Stream it live and print survival time. Does the ball still stay on the plate for 10 seconds?

If the ball falls off, ask Claude to tune the gains:

> The ball survived 4 seconds with force 0.5N. Try increasing Kp or Kd to make the controller react faster. Run several experiments and tell me which gains work best.

### Challenge 2 — Stronger pushes

> Increase the disturbance force to 1.0 Newton. Adjust the gains to compensate. What's the highest force the ball can survive for 10 seconds?

### Challenge 3 (advanced) — Oscillating plate

> Add a slow sinusoidal oscillation to the plate — make the arm rock the plate in a small circle while still keeping the ball centered. Start at 0.5 Hz. What's the fastest oscillation frequency where the ball still survives 10 seconds?

### Bonus — Competition

If time allows, try to beat other participants:

- **Highest disturbance force survived for 10 seconds**
- **Fastest plate oscillation with ball balanced**
- **Longest survival time with both disturbances and oscillation**

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

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| VS Code disconnected | Click the green `><` icon → "Connect to Host" → reconnect |
| Terminal is frozen | Open a new terminal: Terminal → New Terminal |
| Claude is not responding | Press `Ctrl+C` to cancel, then type `claude` to restart |
| Can't see the live video | Click the port forwarding popup in VS Code, or go to the Ports tab and click the globe icon |
| Video shows but is frozen | Tell Claude: "Restart the live stream" |
| `(workshop_env)` not showing | Run: `source ~/workshop_env/bin/activate` |

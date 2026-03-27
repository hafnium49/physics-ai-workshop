# Autonomous Demo Script (Host Guide)

Run this interactively in Claude Code before participants arrive.
Show ONLY the browser stream on the projector — not the terminal.
Total time: 3-5 minutes.

---

## Setup

1. Open VS Code terminal, navigate to the workspace:
   ```
   cd ~/physics_sim
   ```
2. Start Claude Code:
   ```
   claude
   ```
3. Open your browser and have it ready for `http://localhost:8080`

---

## Prompt 1 — Load and Stream

Paste into Claude:

> Load panda_ball_balance.xml. Set the arm to the home pose, place the ball on the plate, and start a live stream on port 8080 using mujoco_streamer.py. Run the simulation in an infinite loop so I can watch the ball fall off naturally.

**What happens:** Claude writes a script, runs it. The stream starts. The ball rolls off the plate in about 1 second.

**Show on projector:** The browser stream. Narrate: *"This is the digital twin. The robot is holding a plate, and we just dropped a ball on it. No control yet — the ball falls off immediately."*

---

## Prompt 2 — First PID Attempt

Press Ctrl+C to stop the simulation, then paste:

> Write a PID controller that tries to keep the ball centered on the plate. Use the ball's position relative to the plate center as the error signal. Pick whichever wrist joints you think control the plate tilt. Stream it live and print "Survival Time" when the ball falls off. Auto-reset and try again.

**What happens:** Claude writes a PID controller. It will likely pick joints that don't work well. Ball falls off in under 2 seconds.

**Narrate:** *"The AI wrote a controller, but the ball still falls off. The AI picked the wrong joints. Watch what happens when we ask it to investigate."*

---

## Prompt 3 — Diagnose

Paste:

> The survival time is very short. Can you check which joints actually move the plate? Try nudging each wrist joint by a small amount and measure how much the plate position changes. Report what you find.

**What happens:** Claude runs a joint authority analysis. It discovers that some joints have strong plate control and others have almost none.

**Narrate:** *"The AI just discovered — by itself — which joints actually matter. This is the 'Autonomous Scientist' at work."*

---

## Prompt 4 — Fix and Succeed

Paste:

> Switch to the joints with the most plate authority and use a positive correction sign. Try again.

**What happens:** Ball balances for 10 seconds. The stream shows the ball sitting steadily on the plate.

**Narrate:** *"Ten seconds. The AI found the right control architecture through trial and error. That's what you'll be doing in the next hour — but YOU decide what to try."*

---

## Prompt 5 — Challenge (optional, if time allows)

Paste:

> Add random force disturbances that push the ball every 2 seconds. Can the controller still keep the ball on the plate?

**What happens:** Ball gets pushed around but stays on the plate.

**Narrate:** *"Even under disturbance, the controller holds. In your session, you'll push the limits — stronger forces, faster plate oscillation."*

---

## After the Demo

1. Press Ctrl+C to stop the simulation
2. Close the Claude Code session: type `/exit`
3. Tell participants: *"Now it's your turn. Follow the guide on your printed card."*

# Artificial Intelligence & Bot Architecture (`src/ai/`)

This directory houses the rule-based automated opponent, modeled as a baseline controller and sparring partner for adversarial Reinforcement Learning.

---

## 1. Directory File Overview

| File | Description |
| :--- | :--- |
| [`heuristic_bot.py`](heuristic_bot.py) | Implementation of the Hierarchical Zone & Intercept State Machine (HZISM) governing the Orange adversarial car. |

---

## 2. Hierarchical Zone & Intercept State Machine (HZISM)

The automated Orange car operates on an asynchronous finite state machine structured around field geometry, ball trajectory prediction, and zone partitioning:

```text
                  +--------------------------------------+
                  |               KICKOFF                |
                  |     (Start / Post-Goal Reset)        |
                  +--------------------------------------+
                                     |
                                     v
                  +--------------------------------------+
                  |               ATTACK                 |
                  |     (Drive & Strike toward Net)      |
                  +--------------------------------------+
                         /           |           \
                        /            |            \
                       v             |             v
          +-------------------+      |      +------------------+
          |    ROTATE_BACK    |      |      |      DEFEND      |
          |  (Shadow Defense) |      |      | (Goal Clearance) |
          +-------------------+      |      +------------------+
                                     v
                            +------------------+
                            |      AERIAL      |
                            | (High Ball Save) |
                            +------------------+
```

---

## 3. State Responsibilities & Decision Logic

### A. `KICKOFF`
* **Trigger**: Activated immediately upon match start or after a goal reset.
* **Behavior**: Commits full forward throttle with continuous rocket boost directly toward the midfield kickoff ball.
* **Kickoff Power Strike**: When closing within $2.9\text{ m}$ of the ball, the bot executes an explosive forward dodge flip to maximize initial momentum transfer toward the opponent's half.

### B. `ROTATE_BACK` (Shadow Defense)
* **Trigger**: Engaged whenever the bot gets caught downfield of the ball ($x_{car} \le x_{ball} - 0.2\text{ m}$).
* **Anti-Own-Goal Recovery**: Driving directly toward the ball from this position would push the ball into its own net. Instead, the bot steers toward a safe recovery waypoint behind the ball toward its own goal post, navigating around the ball before re-engaging.
* **Transition**: Exits back to `DEFEND` or `ATTACK` once the bot is positioned safely on the goal side of the ball ($x_{car} > x_{ball}$).

### C. `DEFEND` (Zone Goalie Routine)
* **Trigger**: Activated when the ball is inside the Orange defensive zone ($x > 22.0\text{ m}$).
* **Ballistic Lookahead**: Computes ball trajectory prediction over lookahead horizon $\Delta t \in [0.1\text{ s}, 0.5\text{ s}]$ incorporating gravity and bounce.
* **Aerial Save Trigger**: If the incoming shot's predicted intercept height exceeds $y = 3.0\text{ m}$, the bot transitions immediately to `AERIAL`.
* **Ground Clearance**: If the ball is low, the bot aligns along the clearance tangent to punch the ball upfield away from the goal mouth.

### D. `AERIAL` (High Ball Intercept)
* **Trigger**: High aerial crosses, elevated rebounds off the vertical deflection wall, or high goalward shots.
* **Launch Targeting**: Computes intercept angle $\theta_{launch} = \operatorname{atan2}(\Delta y, \Delta x)$.
* **Execution**: Fires Jump 1, commands orientation pitch toward $\theta_{launch}$, and applies full rocket boost thrust. Upon closing within $2.0\text{ m}$ of the aerial ball, executes a Jump 2 dodge flip to strike the ball downward or forward.

### E. `ATTACK` (Offensive Drive)
* **Trigger**: Default offensive state when positioned behind the ball on the opponent's side of the pitch.
* **Strike Pathing**: Steers toward the contact point that directs the ball toward the opponent's elevated goal mouth ($x_{left} = 4.5\text{ m}, y \in [4.36, 9.64]\text{ m}$).
* **Dynamic Maneuvers**: Initiates power jumps for bouncing balls and speed flips on open strikes.

---

## 4. Runtime Bot Toggling & Symmetrical State

* **Dynamic Toggle**: The bot can be enabled or disabled at any time during simulation via:
  ```python
  sim.set_orange_enabled(enabled=True, is_bot=True)
  ```
  or by pressing the `B` key in interactive mode.
* **Symmetrical Telemetry**: The bot uses identical input capabilities (`CarAction`: directional vector, jump, boost) and physical parameters as the human / RL player, ensuring a fair, physically grounded adversarial baseline.

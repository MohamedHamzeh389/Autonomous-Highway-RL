# Autonomous Highway Follower: RL Kinematic Environment

A custom Reinforcement Learning environment built using OpenAI Gymnasium and Stable Baselines3. This project simulates longitudinal vehicle control, training an autonomous agent (PPO) to maintain a safe following distance behind a dynamically behaving leader vehicle while optimizing for passenger comfort.

## 🚀 Project Overview

The goal of this environment is to train an AI to act as an Adaptive Cruise Control (ACC) system. The follower car must learn to balance three competing objectives:
1. **Safety:** Avoid rear-end collisions.
2. **Time Headway:** Maintain an optimal 1.5-second gap behind the leader.
3. **Passenger Comfort:** Minimize "jerk" (rapid changes in acceleration) to prevent whiplash.

## 🧠 Environment Physics & State Space

The environment uses a custom 1D kinematic physics engine. The observation space consists of a continuous 3-element array:
* `obs[0]`: Follower Velocity ($v_f$) bounded between $0.0$ and $40.0$ m/s.
* `obs[1]`: Leader Velocity ($v_l$) bounded between $0.0$ and $40.0$ m/s.
* `obs[2]`: Relative Distance ($d$) bounded between $0.0$ and $100.0$ m.

The action space is a continuous 1D array representing the follower's acceleration, clipped between $-5.0$ and $5.0$ m/s².

**Dynamic Leader Behavior:**
To prevent the agent from simply memorizing a static speed, the leader vehicle acts as an unpredictable human driver, updating its velocity every step using a uniform random distribution: `np.random.uniform(-5.0, 5.0)`.

## 🧮 Reward Function & The "Cliff Problem"

The agent's reward is calculated primarily based on the Time Headway error and a Jerk penalty. 

$$TimeHeadway = \frac{Distance}{v_f + 1e-5}$$
$$Error = |TimeHeadway - 1.5|$$
$$Jerk = |a_t - a_{t-1}|$$

**Overcoming the RL "Cliff Problem":**
During initial training (50,000 timesteps), the agent exhibited a classic RL phenomenon: it preferred a "comfortable crash" over evasive braking because the Jerk penalty heavily discouraged slamming the brakes. The agent was experiencing the "Cliff Problem"—receiving high rewards for maintaining a good headway right up until the millisecond it crashed, failing to anticipate the danger.

**The Solution:**
1. Implemented a strict $-100$ terminal crash penalty.
2. Scaled the training to **500,000 timesteps**, allowing the PPO algorithm enough environmental interaction to look past the immediate Jerk penalty and learn true evasive braking. The final model successfully decelerates to match erratic leader behavior without crashing.

## ⚙️ Dependencies & Installation

To run this environment, you will need the following Python libraries:

* `gymnasium`
* `stable-baselines3`
* `numpy`

## 📊 Results and Terminal
After 500,000 timesteps of training, the agent successfully converges. It learns to smoothly modulate its acceleration, maintaining a ~1.5s headway and prioritizing collision avoidance over passenger comfort only when absolutely necessary (e.g., when the leader vehicle initiates a hard brake).

<img width="1043" height="315" alt="image" src="https://github.com/user-attachments/assets/339ae35c-ca69-44c4-bb5b-933ecd9ebd77" />

# Autonomous Highway Follower: RL Kinematic Environment

A custom Reinforcement Learning environment built using OpenAI Gymnasium and Stable Baselines3. This project simulates longitudinal vehicle control, training an autonomous agent (PPO) to maintain a safe following distance behind a dynamically behaving leader vehicle using real-world highway traffic data while optimizing for passenger comfort.

## 🚀 Project Overview

The goal of this environment is to train an AI to act as an Adaptive Cruise Control (ACC) system. The follower car must learn to balance three competing objectives:
* **Safety:** Avoid rear-end collisions by respecting Time to Collision (TTC) minimums.
* **Time Headway:** Maintain an optimal 1.5-second gap behind the leader.
* **Passenger Comfort:** Minimize "jerk" (rapid changes in acceleration) to prevent whiplash.

## 🧠 Environment Physics & State Space

The environment uses a custom 1D kinematic physics engine. The observation space consists of a continuous 3-element array:
* `obs[0]`: Follower Velocity (v_f) bounded between 0.0 and 40.0 m/s.
* `obs[1]`: Leader Velocity (v_l) bounded between 0.0 and 40.0 m/s.
* `obs[2]`: Relative Distance (d) bounded between 0.0 and 100.0 m.

The action space is a continuous 1D array representing the follower's acceleration, clipped between −5.0 and 5.0 m/s².

**Dynamic Initialization:** To ensure the agent is evaluated on survivable edge cases, the follower's starting speed is dynamically bounded to spawn within ±5.0 m/s of the leader's starting speed, preventing physically impossible crash scenarios on step 1.

**Real-World Leader Behavior:** Instead of basic random uniform speed changes, the leader vehicle uses telemetry from the NGSIM I-80 dataset. The agent must adapt to the erratic, unpredictable speed changes of actual human drivers.

## 🧮 Reward Function 

The agent's reward is calculated primarily based on the Time Headway error and a Jerk penalty.
* Time Headway = Distance / (v_f + 1e-5)
* Error = |Time Headway - 1.5|
* Jerk = |a_t - a_{t-1}|


## ⚙️ Dependencies & Installation

To run this environment and view the telemetry dashboards, you will need the following Python libraries:
* `gymnasium`
* `stable-baselines3`
* `numpy`
* `matplotlib`

## 📊 Results and Telemetry

After 500,000 timesteps of training, the agent successfully converges. It learns to smoothly modulate its acceleration, absorbing the erratic driving of the human leader to prioritize passenger comfort while flawlessly maintaining the ~1.5s headway. 

A Matplotlib telemetry dashboard is included in the testing loop to visually prove the physics engine and safety metrics. As shown in the safety subplots, the agent successfully keeps the TTC well above dangerous thresholds even during closure events.

<img width="975" height="611" alt="image" src="https://github.com/user-attachments/assets/82d01df4-6ee8-448f-bfc7-01dbfae4da32" />

<img width="975" height="458" alt="image" src="https://github.com/user-attachments/assets/32befee7-a24c-4922-99b7-fe1f5d43c2d8" />

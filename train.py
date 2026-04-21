import os
import numpy as np
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from stable_baselines3 import PPO
from custom_highway import HighwayFollowerEnv

env = HighwayFollowerEnv()


print("Starting training...")
#model = PPO("MlpPolicy", env, verbose=1)
#model.learn(total_timesteps=1000000, progress_bar=True)
#model.save("PPO_HighwayFollower")
#del model


obs, info = env.reset()
model = PPO.load("PPO_HighwayFollower")

done = False
step_count = 0

times = []
follower_speeds = []
leader_speeds = []
distances = []

print("Starting simulation...")

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    done = terminated or truncated

    current_time = (step_count + 1) * 0.1 

    times.append(current_time)
    follower_speeds.append(obs[0])
    leader_speeds.append(obs[1])
    distances.append(obs[2])

    print(f"Sec {current_time:.1f} | Follower: {obs[0]:.1f}m/s | Leader: {obs[1]:.1f}m/s | Dist: {obs[2]:.1f}m | Reward: {reward:.2f}")

    if terminated:
          print(" CRASHED! Distance hit 0. ")
    elif truncated:
            print(" SUCCESS! Survived safely!")
    step_count += 1

ttc_values = []
for i in range(len(distances)):
    delta_v = follower_speeds[i] - leader_speeds[i]
    d = max(distances[i], 0.0)

    if delta_v > 0.1 and d > 0:
        ttc = d / delta_v
        ttc = min(ttc, 15.0)
    else:
        ttc = 15.0

    ttc_values.append(ttc)

plt.figure(figsize=(10, 6))

# Top Graph: Speeds
plt.subplot(2, 1, 1) 
plt.plot(times, follower_speeds, label="Follower Speed ", color='blue', linewidth=2)
plt.plot(times, leader_speeds, label="Leader Speed", color='orange', linewidth=2, linestyle='--')
plt.title("Cruise Control Highway Follower Performance")
plt.ylabel("Speed (m/s)")
plt.legend()
plt.grid(True)

# Bottom Graph
plt.subplot(2, 1, 2) 
# plt.plot(times, distances, label="Distance", color='green', linewidth=2)
plt.plot(times, np.divide(distances, follower_speeds), label="Time Headway", color='green', linewidth=2)
plt.axhline(y=1.5, color='green', linestyle=':', label="Target THW (1.5s)")
plt.plot(times, ttc_values, label="Time to Collision (TTC)", color='red', linewidth=2, linestyle='--')
plt.title("Safety Metrics")
plt.xlabel("Time (Seconds)")
plt.ylabel("Time (s)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
        
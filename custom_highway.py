import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

class HighwayFollowerEnv(gym.Env):
    
    def __init__(self):
        super(HighwayFollowerEnv, self).__init__()
        
        
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)

       
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([40.0, 40.0, 100.0]),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        
        self.state = np.array([20.0, 20.0, 30.0], dtype=np.float32)
        info = {}
        
        self.previous_acceleration = 0.0

        return self.state, info

    def step(self, action):
        acceleration = action[0]
        self.state[0] = np.clip(self.state[0] + acceleration, 0.0, 40.0)

        self.state[2] = np.clip((self.state[1] - self.state[0]) + self.state[2], 0.0, 100.0)

        self.state[1] += self.np_random.uniform(-5.0, 5.0)
        self.state[1] = np.clip(self.state[1], 0.0, 40.0)

        terminated = self.state[2] == 0
        
        headway = self.state[2] /(self.state[0] + 1e-5)

        error = abs(headway - 1.5)

        jerk = abs(acceleration - self.previous_acceleration)
        self.previous_acceleration = acceleration

        
            
        if terminated:
            reward = -100
        elif error > 10 and jerk > 5:
            reward = -15
        else: reward = 15 - error - jerk
            
        truncated = False 
        info = {}

        return self.state, reward, terminated, truncated, info
    
    # Execution section:
env = HighwayFollowerEnv()

model = PPO("MlpPolicy", env, verbose=1)

print("Starting training...")

model.learn(total_timesteps= 500000, progress_bar=True)

obs, info = env.reset()

for i in range(20):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Sec {i+1:02d} | Follower: {obs[0]:.1f}m/s | Leader: {obs[1]:.1f}m/s | Dist: {obs[2]:.1f}m | Reward: {reward:.2f}")
    
    if terminated:
        print(" CRASHED! Distance hit 0. ")
        break
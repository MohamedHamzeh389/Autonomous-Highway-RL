import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pickle
import random
from stable_baselines3 import PPO

class HighwayFollowerEnv(gym.Env):
    
    def __init__(self):
        super(HighwayFollowerEnv, self).__init__()
        
        self.dt = 0.1
        
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)
        
       
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([40.0, 40.0, 100.0]),
            dtype=np.float32 )

           # Load the  real-world values
        with open("test_data", "rb") as f:
            self.trajectories = pickle.load(f)

        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Pick a random human trajectory from the dataset
        self.current_trajectory = random.choice(self.trajectories)

        
        self.current_step = 0 

        leader_start_speed = self.current_trajectory[self.current_step]

        min_speed = max(0.0, leader_start_speed - 5.0)
        max_speed = leader_start_speed + 5.0

        follower_start_speed = random.uniform(min_speed, max_speed)
        start_distance = random.uniform(40.0, 60.0)

        
        self.state = np.array([follower_start_speed, leader_start_speed, start_distance], dtype=np.float32)
        
    
        
        self.previous_acceleration = 0.0

        info = {}
        return self.state, info

    def step(self, action):
        acceleration = action[0]
        self.current_step += 1

        # Check how long this specific driver's data is
        traj_length = len(self.current_trajectory)

        #check if we ran out of data
        if self.current_step >= traj_length - 1:
            # Grab the very last speed
            self.state[1] = self.current_trajectory[-1] 
            # Tell the AI we ran out of time/data
            truncated = True  
        else:
            self.state[1] = self.current_trajectory[self.current_step]
            truncated = False


        self.state[1] = np.clip(self.state[1], 0.0, 40.0)

        self.state[0] = np.clip(self.state[0] + (acceleration*self.dt), 0.0, 40.0)

        self.state[2] = np.clip((self.state[1] - self.state[0])*self.dt + self.state[2], 0.0, 100.0)

        
        

        terminated = self.state[2] <= 0
        
        headway = self.state[2] /(self.state[0] + 1e-5)

        error = abs(headway - 1.5)

        jerk = abs(acceleration - self.previous_acceleration)/(self.dt + 1e-5)
        self.previous_acceleration = acceleration

        
            
        if terminated:
            reward = -100
        else: reward = 15 - (error*2) - (jerk*0.1)
            
        

        
        
        info = {}

        return self.state, reward, terminated, truncated, info
    
    
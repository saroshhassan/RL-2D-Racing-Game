
#view_training.py/visualize the trained car

import time
import sys
import os
from stable_baselines3 import PPO
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from env import car_race_env_lidar


def view_training():
    print("Loading model and starting visualization...")
    env = car_race_env_lidar.CarRaceEnvLidar(render_mode="human")
    
    try:
        model = PPO.load("models/pyrace_cpu_ppo")
        print("Model loaded successfully!")
    except:
        print("No trained model found. Using random actions...")
        model = None
    
    episodes = 10
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"Episode {episode + 1}")
        
        while not done:
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
                
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            env.render()
            time.sleep(0.03)  # Small delay
            
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")
        
        if 'lap_completed' in info:
            print("LAP COMPLETED!")
        elif 'crashed' in info:
            print("CRASHED!")
    
    env.close()

if __name__ == "__main__":
    view_training()

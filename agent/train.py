#train loop
import os
import csv
import time
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import subproc_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from env import car_race_env_lidar 
# ------------------------
# Custom Callback for Logging & Checkpoints
# ------------------------
class RaceLoggerCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, log_path: str, verbose=1):
        super(RaceLoggerCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.log_path = log_path
        self.episode_rewards = []
        self.episode_lengths = []

        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # prepare CSV header
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestep", "episode", "total_reward", "episode_length", 
                                "agent_hp", "opponent_hp", "lap_completed", "crashed"])

    def _on_step(self) -> bool:
        if self.num_timesteps % 5 == 0:
            # Access the underlying environment and render it
            try:
                # Get the base environment from the vectorized wrapper
                base_env = self.training_env.envs[0].env  # DummyVecEnv -> Monitor -> CarRaceEnv
                while hasattr(base_env, 'env'):  # Unwrap if needed
                    base_env = base_env.env
                
                base_env.render()
                time.sleep(0.02)  # Small delay to see what's happening
                
            except Exception as e:
                if self.verbose:
                    print(f"Render failed: {e}")
                    
                    
                    ##^^38-51 render env
        # Get info from the environment
        infos = self.locals.get("infos", [{}])
        dones = self.locals.get("dones", [False])

        rewards = self.locals.get("rewards", [0])
        
        for i, (info, done, reward) in enumerate(zip(infos, dones, rewards)):
            if done and info:
                agent_hp = info.get("agent_hp", 0)
                opp_hp = info.get("opponent_hp", 0)
                lap_completed = info.get("lap_completed", False)
                crashed = info.get("crashed", False)
                
                # Log episode data
                episode_length = info.get("episode", {}).get("l", 0)
                episode_reward = info.get("episode", {}).get("r", 0)
                
                with open(self.log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.num_timesteps, 
                        len(self.episode_rewards),
                        episode_reward,
                        episode_length,
                        agent_hp, 
                        opp_hp, 
                        lap_completed, 
                        crashed
                    ])
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                if self.verbose and len(self.episode_rewards) % 10 == 0:
                    avg_reward = sum(self.episode_rewards[-10:]) / 10
                    print(f"Episode {len(self.episode_rewards)}, Avg Reward (last 10): {avg_reward:.2f}")

        # save checkpoints
        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            save_file = os.path.join(self.save_path, f"ppo_checkpoint_{self.num_timesteps}")
            self.model.save(save_file)
            if self.verbose:
                print(f"Checkpoint saved: {save_file}")

        return True

# ------------------------
# Train
# ------------------------
def make_env():
    def _init():
        env = car_race_env_lidar.CarRaceEnvLidar(render_mode="human")
        env = Monitor(env)   # ✅ Monitor first
        return env
    return _init

# Wrap in DummyVecEnv
env = DummyVecEnv([make_env()])

# Apply VecNormalize
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

if __name__ == "__main__":
    print("Starting Car Race AI Training...")
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Create vectorized environment
    #env = DummyVecEnv([make_env])

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        batch_size=512,
        n_steps=4096,
        learning_rate=3e-4,
        ent_coef=0.1,  # Encourage exploration
        clip_range=0.2,
        tensorboard_log="./tb_logs"  # for tensorboard monitoring
    )

    # Create callback
    callback = RaceLoggerCallback(
        save_freq=50000,  # checkpoint every 50k steps
        save_path="models/checkpoints",
        log_path="logs/training_log.csv",
        verbose=1
    )

    print("Starting training...")
    print(f"Total timesteps: 200,000")
    print(f"Checkpoints every 50,000 steps")
    print(f"Logs saved to: logs/training_log.csv")

    try:
        model.learn(total_timesteps=200_000, callback=callback, progress_bar=True)
        
        # Save final model
        final_model_path = "models/pyrace_cpu_ppo"
        model.save(final_model_path)
        print(f" Final model saved at {final_model_path}")
        
        # Save final checkpoint
        model.save("models/checkpoints/ppo_final")
        print(" Final checkpoint saved")
        
    except KeyboardInterrupt:
        print("\n Training interrupted by user")
        model.save("models/checkpoints/ppo_interrupted")
        print(" Model saved before exit")
    
    except Exception as e:
        print(f" Training failed with error: {e}")
        model.save("models/checkpoints/ppo_error")
        print(" Model saved after error")
    
    finally:
        env.close()
        print(" Training session ended")
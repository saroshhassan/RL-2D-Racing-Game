# env/car_race_env.py- game wrapped in an environment for the agent to play/train on
import math
import numpy as np
import pygame
import os, sys
from gymnasium import Env, spaces
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from game import car, track
from game.collision import check_collision, check_boundaries, check_boundary_collision
from config import SCREEN_HEIGHT, SCREEN_WIDTH


class CarRaceEnv(Env):
    """
    Gym environment where the agent controls the CPU car.
    The other car is a scripted opponent (simple waypoint/heuristic).
    Observations are a vector (agent x,y,angle,vx,vy, opponent x,y,angle,vx,vy, agent hp, opp hp).
    Actions: [steer (-1..1), accel (-1..1)]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    

    def __init__(self, render_mode=None):
        super().__init__()
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        self.render_mode = render_mode
        self.finish_point=(1042,150)

        # Continuous 2-d action: steer, throttle
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # observation: agent x,y,angle,speed, opp x,y,angle,speed, agent_hp, opp_hp
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(10,), dtype=np.float32)

        # Initialize pygame for both rendering and headless modes
        pygame.init()
        if render_mode == "human":
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
        else:
            # Headless mode for training - no display
            self.screen = pygame.Surface((self.width, self.height))
            self.clock = pygame.time.Clock() if render_mode else None

        # Internal state placeholders
        self.track = track.Track()
        self.agent = car.Car(90, 715, image_path="assets/car/car2.png", is_cpu=True)  # AI car
        self.opponent = car.Car(90, 625, image_path="assets/car/car1.png", is_cpu=False)  # Scripted opponent
        self.max_speed = 8.0

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        # create cars: agent is AI-controlled CPU car, opponent is scripted
        # Use same spawn positions as in core.py but agent gets the CPU position
        self.agent = car.Car(90, 715, image_path="assets/car/car2.png", is_cpu=True)  # AI car
        self.opponent = car.Car(90, 625, image_path="assets/car/car1.png", is_cpu=False)  # Scripted opponent

        # Reset health
        self.agent.health = 100
        self.opponent.health = 100
        
        # Reset timers
        if hasattr(self.agent, 'timer'):
            self.agent.timer.reset()
        if hasattr(self.opponent, 'timer'):
            self.opponent.timer.reset()

        self.steps = 0
        return self._get_obs(), {}

    """def _get_obs(self):
        a = self.agent
        o = self.opponent
        
        # normalize positions to [-1,1], angles as sin/cos, speeds relative
        obs = np.array([
            (a.rect.centerx / self.width - 0.5) * 2,
            (a.rect.centery / self.height - 0.5) * 2,
            math.sin(math.radians(a.angle)),
            math.cos(math.radians(a.angle)),
            a.speed / self.max_speed,
            (o.rect.centerx / self.width - 0.5) * 2,
            (o.rect.centery / self.height - 0.5) * 2,
            math.sin(math.radians(o.angle)),
            math.cos(math.radians(o.angle)),
            o.speed / self.max_speed
        ], dtype=np.float32)
        return obs"""#this observation func takes the opponent dir as input for observation
    def _get_obs(self):
        """Returns observation of agent by raycast"""
        a = self.agent

        # --- Car state ---
        obs = [
            (a.rect.centerx / self.width - 0.5) * 2,   # normalized X position
            (a.rect.centery / self.height - 0.5) * 2,  # normalized Y position
            math.sin(math.radians(a.angle)),           # heading (sin)
            math.cos(math.radians(a.angle)),           # heading (cos)
            a.speed / self.max_speed                   # normalized speed
        ]

        # --- Track-relative info ---
        # Sample distances from car center in several directions (like "virtual sensors")
        sensor_angles = [-45, -20, 0, 20, 45]  # degrees relative to car heading
        max_range = 300                         # how far sensors can "see"
        
        for sa in sensor_angles:
            # Absolute angle in radians
            ray_angle = math.radians(a.angle + sa)
            dx = math.cos(ray_angle)
            dy = math.sin(ray_angle)

            dist = max_range
            for d in range(1, max_range, 5):  # step ray outward
                x = int(a.rect.centerx + dx * d)
                y = int(a.rect.centery + dy * d)

                if not (0 <= x < self.width and 0 <= y < self.height):
                    dist = d
                    break

                # Check collision with boundary (off-road pixels)
                if self.boundary_mask.get_at((x, y)) == 1:  # assuming mask is binary
                    dist = d
                    break

            obs.append(dist / max_range)  # normalize

        return np.array(obs, dtype=np.float32)


    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        steer, accel = float(action[0]), float(action[1])

        # Convert actions to key presses for the agent
        keys = self._action_to_keys(steer, accel)
        
        # Apply action to agent using the move method (like in core.py)
        self.agent.move(keys)

        # Simple scripted opponent behavior using move method
        opponent_keys = self._get_opponent_keys()
        self.opponent.move(opponent_keys)

        reward = 0.0
        done = False
        info = {
            "agent_hp": self.agent.health,
            "opponent_hp": self.opponent.health
        }

        # Use the same collision system as core.py
        # Car-to-car collision
        if check_collision(self.agent, self.opponent):
            self.agent.collide()
            self.opponent.collide()
            reward -= 3.0

        # Track boundary collisions
        if check_boundary_collision(self.agent, self.track):
            self.agent.collide()
            reward -= 2.0
        
        if check_boundary_collision(self.opponent, self.track):
            self.opponent.collide()

        # Check boundaries for start/finish line rewards
        agent_status = check_boundaries(self.agent, self.track)
        if agent_status == "start":
            if hasattr(self.agent, 'timer') and not self.agent.timer.running:
                self.agent.timer.start()
                reward += 2.0  # Reward for crossing start line
        elif agent_status == "finish":
            if hasattr(self.agent, 'timer') and self.agent.timer.running:
                self.agent.timer.stop()
                reward += 50.0  # Big reward for finishing lap
                done = True
                info["lap_completed"] = True

        # Rewards for good behavior
        if self.agent.health > 10:
            reward += 0.001  # Staying alive
            if self.agent.health > 50:
                reward+=.002
        reward += self.agent.speed * 5  # Moving forward

        # Penalties
        if self.agent.health <= 0:
            reward -= 20.0
            done = True
            info["crashed"] = True

        self.steps += 1
        
        # Prevent infinite episodes
        if self.steps > 1000:
            done = True
            info["timeout"] = True

        return self._get_obs(), float(reward), bool(done), False, info

    def _action_to_keys(self, steer, accel):
        """Convert continuous actions to discrete key presses"""
        keys = {}
        
        # Acceleration/braking
        if accel > 0.1:
            keys[pygame.K_UP] = True
        elif accel < -0.1:
            keys[pygame.K_DOWN] = True
            
        # Steering
        if steer < -0.1:
            keys[pygame.K_LEFT] = True
        elif steer > 0.1:
            keys[pygame.K_RIGHT] = True
            
        return keys

    def _get_opponent_keys(self):
        """Simple scripted opponent behavior"""
        import random
        
        keys = {}
        
        # Always try to accelerate
        keys[pygame.K_UP] = True
        
        # Random steering occasionally
        if random.random() < 0.05:
            if random.random() < 0.5:
                keys[pygame.K_LEFT] = True
            else:
                keys[pygame.K_RIGHT] = True
                
        return keys

    def render(self):
        if self.render_mode is None:
            return None
            
        if self.screen is None:
            return None
            
        # Fill background
        self.screen.fill((0, 150, 0))
        
        # Draw track
        self.track.draw(self.screen)
        
        # Draw cars
        self.agent.draw(self.screen)
        self.opponent.draw(self.screen)
        
        if self.render_mode == "human":
            pygame.display.flip()
            if self.clock:
                self.clock.tick(30)
                
        return None

    def close(self):
        if hasattr(self, 'screen') and self.screen is not None:
            pygame.quit()
# env/car_race_env_lidar.py
import math
import numpy as np
import pygame
from gymnasium import Env, spaces
from game import car, track
from game.collision import check_collision, check_boundary_collision
from config import SCREEN_HEIGHT, SCREEN_WIDTH

class CarRaceEnvLidar(Env):
    """
    Car racing environment with LIDAR-based observations
    and checkpoint progression system.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.width, self.height = SCREEN_WIDTH, SCREEN_HEIGHT
        self.render_mode = render_mode

        # Continuous action: steer (-1..1), throttle (-1..1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation: [car_state + lidar_distances + lidar_hit_flags]
        # car_state = [x, y, sin(angle), cos(angle), speed, time_elapsed]
        # lidar: n_beams * 2 → (distance, hit_flag)
        self.n_beams = 13
        obs_dim = 6 + self.n_beams * 2
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        # Init pygame
        pygame.init()
        if render_mode == "human":
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
        else:
            self.screen = pygame.Surface((self.width, self.height))
            self.clock = pygame.time.Clock() if render_mode else None

        # Track and cars
        self.track = track.Track()
        self.agent = car.Car(90, 715, image_path="assets/car/car2.png", is_cpu=True)
        self.max_speed = 8.0

        # Checkpoint system
        self.total_cp = [(950, 550), (250, 350), (1042, 150)]  # stack of checkpoints
        self.overtaken_cp = []
        self.current_target = self.total_cp[0]
        self.check_finish = False

    # -------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent = car.Car(90, 715, image_path="assets/car/car2.png", is_cpu=True)
        self.agent.health = 100
        if hasattr(self.agent, "timer"):
            self.agent.timer.reset()

        self.total_cp = [(950, 550), (250, 350), (1042, 150)]
        self.overtaken_cp = []
        self.current_target = self.total_cp[0]
        self.check_finish = False
        self.steps = 0

        return self._get_obs(), {}

    # -------------------------------
    def _get_obs(self):
        """Return state vector: [car state + lidar]"""
        a = self.agent
        obs = [
            (a.rect.centerx / self.width - 0.5) * 2,
            (a.rect.centery / self.height - 0.5) * 2,
            math.sin(math.radians(a.angle)),
            math.cos(math.radians(a.angle)),
            a.speed / self.max_speed,
            self.agent.timer.get_time() / 100.0  # normalize time
        ]

        # LIDAR beams
        sensor_angles = np.linspace(-90, 90, self.n_beams)  # relative to heading
        max_range = 300

        for sa in sensor_angles:
            ray_angle = math.radians(a.angle + sa)
            dx, dy = math.cos(ray_angle), math.sin(ray_angle)

            dist = max_range
            hit = 0
            for d in range(1, max_range, 3):
                x = int(a.rect.centerx + dx * d)
                y = int(a.rect.centery + dy * d)

                if not (0 <= x < self.width and 0 <= y < self.height):
                    dist, hit = d, 1
                    break
                if self.track.boundary_mask.get_at((x, y)) == 1:
                    dist, hit = d, 1
                    break

            obs.append(dist / max_range)  # normalized distance
            obs.append(hit)               # 1 = collision boundary, 0 = free

        return np.array(obs, dtype=np.float32)

    # -------------------------------
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        steer, accel = float(action[0]), float(action[1])

        # Apply action
        keys = self._action_to_keys(steer, accel)
        self.agent.move(keys)

        reward, done = 0.0, False
        info = {}

        # Collision penalty
        if check_boundary_collision(self.agent, self.track):
            self.agent.collide()
            reward -= 3.0

        # Checkpoint progression
        if self.current_target:
            dist = self.agent.calculate(self.current_target)  # uses same formula
            if dist < 250:
                reward += 20.0
                self.overtaken_cp.append(self.current_target)
                self.total_cp.pop(0)
                if self.total_cp:
                    self.current_target = self.total_cp[0]
                else:
                    self.current_target = None
                    self.check_finish = True

        # Finish reward
        if self.check_finish and not self.total_cp:
            reward += 100.0
            done = True
            info["lap_completed"] = True

        # Alive reward
        reward += 0.01
        reward += self.agent.speed * 0.5  # encourage speed

        if self.agent.health <= 0:
            reward -= 20.0
            done = True
            info["crashed"] = True

        self.steps += 1
        if self.steps > 2000:
            done = True
            info["timeout"] = True

        return self._get_obs(), float(reward), bool(done), False, info

    # -------------------------------
    def _action_to_keys(self, steer, accel):
        keys = {}
        if accel > 0.1:
            keys[pygame.K_UP] = True
        elif accel < -0.1:
            keys[pygame.K_DOWN] = True
        if steer < -0.1:
            keys[pygame.K_LEFT] = True
        elif steer > 0.1:
            keys[pygame.K_RIGHT] = True
        return keys

    # -------------------------------
    def render(self):
        if self.render_mode is None:
            return None
        self.screen.fill((0, 150, 0))
        self.track.draw(self.screen)
        self.agent.draw(self.screen)

        # Debug: draw checkpoints
        for cp in self.total_cp:
            pygame.draw.circle(self.screen, (0, 0, 255), cp, 10)
        for cp in self.overtaken_cp:
            pygame.draw.circle(self.screen, (0, 255, 0), cp, 10)

        if self.render_mode == "human":
            pygame.display.flip()
            if self.clock:
                self.clock.tick(30)
        return None

    def close(self):
        pygame.quit()

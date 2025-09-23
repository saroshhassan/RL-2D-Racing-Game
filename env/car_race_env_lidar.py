# env/car_race_env_lidar.py
import math
import numpy as np
import pygame
from gymnasium import Env, spaces
from game import car, track
from game.collision import check_collision, check_boundary_collision, check_boundaries
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
        obs_dim = 8 + self.n_beams * 2
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
        self.agent2= car.Car(90, 625,image_path= "assets/car/car1.png", is_cpu=True) 
        self.max_speed = 8.0

        # Checkpoint system
        self.total_cp = [(950, 550), (250, 350), (1042, 150)]  # stack of checkpoints
        self.overtaken_cp = []
        self.current_target = self.total_cp[0]
        self.last_dist_to_target = None
        self.check_finish = False
        

    # -------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent = car.Car(90, 715, image_path="assets/car/car2.png", is_cpu=True)
        self.agent.health = 100
        if hasattr(self.agent, "timer"):
            self.agent.timer.reset()
            #debug:vvvvv
            self.agent.timer.start()

        self.total_cp = [(950, 550), (250, 350), (1042, 150)]
        self.overtaken_cp = []
        self.current_target = None       
        self.last_dist_to_target=None
        self.check_finish = False
        self.steps = 0

        return self._get_obs(), {}

    # -------------------------------
    def _get_obs(self,opp_agent=None):
        """Return state vector: [car state + lidar]"""
        a = self.agent
        obs = [
            (a.rect.centerx / self.width - 0.5) * 2,
            (a.rect.centery / self.height - 0.5) * 2,
            math.sin(math.radians(a.angle)),
            math.cos(math.radians(a.angle)),
            (a.speed / self.max_speed)-.5*2,
            a.timer.get_time() / 100000.0,  # normalize time
            float(self.last_dist_to_target or 0.0) / 1000.0,
            a.heading_diff/(2*math.pi) #normalize by 2pi
        
        ]
        opp_agent=opp_agent

        # LIDAR beams
        sensor_angles = np.linspace(-90, 90, self.n_beams)  # relative to heading
        max_range = 300

        for sa in sensor_angles:
            ray_angle = math.radians(a.angle + sa)
            dx, dy = math.cos(-ray_angle), math.sin(-ray_angle)

            dist = max_range
            hit = 0
            for d in range(1, max_range, 3):
                x = int(a.rect.centerx + dx * d)
                y = int(a.rect.centery + dy * d)

                if not (0 <= x < self.width and 0 <= y < self.height): #bound detection
                    dist, hit = d, 1
                    break
                if self.track.boundary_mask.get_at((x, y)) == 1:
                    dist, hit = d, 1
                    break
                
                # Opponent collision check
                if opp_agent:
                    opponent_rect=opp_agent.rect
                    opponent_mask=opp_agent.mask
                    
                        # Check if the ray point is within the opponent's rect
                    if opponent_rect.collidepoint(x, y):
                            # Convert screen coordinates to opponent's local mask coordinates
                        local_x = x - opponent_rect.x
                        local_y = y - opponent_rect.y
                            
                            # Make sure the local coordinates are within mask bounds
                        if (0 <= local_x < opponent_mask.get_size()[0] and 
                            0 <= local_y < opponent_mask.get_size()[1]):
                            if opponent_mask.get_at((local_x, local_y)) == 1:
                                dist,hit = d,2
                                break
            

            obs.append(dist / max_range)  # normalized distance
            obs.append(hit/2)               # 2=opponent car, 1 = collision boundary, 0 = free
        
            
            
        #obs= np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)     
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

        # Get LIDAR data for navigation rewards
        lidar_distances, lidar_hits = self._get_lidar_data()
        
        # LIDAR-based navigation rewards
        #lidar_reward = self._calculate_lidar_reward(lidar_distances, lidar_hits, steer)
        collision_reward=self._calculate_collision_reward()
        #reward += lidar_reward
        reward+=collision_reward
        progress_reward=self._calculate_progress_reward()
        reward+=progress_reward
        time_penalty=self._calculate_time_penalty()
        reward+=time_penalty
        speed_reward=self._calculate_speed_reward()
        reward+=speed_reward    
        heading_reward=self._calculate_heading_reward()
        reward+=heading_reward
        #survival_reward=self._calculate_survival_reward()
        #reward+=survival_reward
        
        
         # Finish reward
        if self.check_finish and not self.total_cp:
            done = True
            self.info["lap_completed"] = True
            self.reset()

        if self.agent.health <= 0:
            done = True
            info["crashed"] = True
            self.reset()
            

        self.steps += 1
        if self.steps > 2000:
            done = True
            info["timeout"] = True
            
            
        if self.steps % 100 == 0:
            print(f"Step {self.steps}: Reward components - "
                  #f"Lidar: {lidar_reward:.2f}, 
                  f"Time: {time_penalty:.2f}, "
                  f"Progress: {progress_reward:.2f}, "
                  #Survival: {survival_reward:.2f},
                  f"Collision: {collision_reward:.2f},"
                  f"Speed: {speed_reward:.2f}"
                  f"Heading :{heading_reward:2f},"
                  f"Total: {reward:.2f}")

        return self._get_obs(), float(reward), bool(done), False, info
    
    def _calculate_heading(self):
        # In step():
        """
        Proof for heading angle:
        CP at (x,y), car at (cx,cy)
        Δx=x-cx
        Δy=y-cy
        tanθ=(Δx/Δy), but
        in PyGame y axis in inverted
        therefore
        -(y-cy)=cy-y
        Δy=cy-y
        tanθ=Δx/Δy
        heading θ= arctan 2(Δy,Δx)
        
        car at θ₁
        heading at θ₂
        
        θ₂-θ₁= angle to steer towards
        
        if θ₁ = n rad and θ₂=0 rad (CP lies on line from (cx,cy) to (x,y))
        θ₂-θ₁=(n+π)/2π -> maintains within circle
        
              
        
        """
        target_dx = self.current_target[0] - self.agent.rect.centerx
        target_dy = self.current_target[1] - self.agent.rect.centery
        target_angle = math.atan2(-target_dy, target_dx)  # careful with pygame's y-axis

        car_angle_rad = math.radians(self.agent.angle)
        heading_diff = abs((target_angle - car_angle_rad + np.pi) % (2*np.pi) - np.pi)
        self.agent.heading_diff=heading_diff
        
        
        return heading_diff

    
    #--------------heading reward
    def _calculate_heading_reward(self):
        
        heading_reward = (1 -self._calculate_heading() / np.pi) * 2.0  # normalized, max at facing target
        
        
        return heading_reward

    #------------speed reward
    def _calculate_speed_reward(self):
        speed_reward=0
        
        # Speed reward (encourage forward movement)
        if self.agent.speed==0:
            speed_reward += self.agent.speed -.5
            
        if self.agent.speed<0:
            speed_reward-=self.agent.speed-1
            
        else:
            speed_reward +=(self.agent.speed/self.max_speed)*4
            
        
       
        
        return speed_reward
    
    def _calculate_survival_reward(self):
        survival_reward=0
        
        if self.agent.health <= 0:
            survival_reward -= 10.0
            
        if self.agent.health>5:
             # Alive reward
            survival_reward += 0.001
            
        return survival_reward
            
        
    
    #------------timespent reward
    
    def _calculate_time_penalty(self):
        max_time=100000.0 
        time_penalty=0
        
        if not self.agent.timer.running:
            time_penalty-=.05
        
        if self.agent.timer.running:
            #timespent reward-incentivize moving faster to get to the finish
            time_penalty-=0.01
            
        return time_penalty
        
    #------------get progress reward
    def _calculate_progress_reward(self):
        # Checkpoint progression (existing code)
        progress_reward=0
        self.current_target = self.total_cp[0]
        if self.current_target:
            
            dist = self.agent.calculate(self.current_target)
            if self.last_dist_to_target is None:
                self.last_dist_to_target = dist
            # Calculate reward based on change in distance
            distance_delta = self.last_dist_to_target - dist
            progress_reward += distance_delta * 5 # Scale it down
        
            self.last_dist_to_target = dist # Update for next step
                
                

            if dist < 30:
                progress_reward+= 5.0
                self.overtaken_cp.append(self.current_target)
                self.total_cp.pop(0)
                if self.total_cp:
                    self.current_target = self.total_cp[0]
                else:
                    self.current_target = None
                    self.check_finish = True

        # Finish reward
        if self.check_finish and not self.total_cp:
            progress_reward+= 10.0
            
            
        return progress_reward
    
    #------------get collision reward
    def _calculate_collision_reward(self):
        # Collision penalty
        collision_reward=0
        if check_boundary_collision(self.agent, self.track):
            self.agent.collide()
            collision_reward-= 3.0  # Increased penalty for hitting walls
            
        
            
        if check_boundaries(self.agent,self.track) == "start": #reward to start the game
            if self.agent.start_flag==0:
                self.agent.start_flag=1
                #self.agent.timer.start()
                collision_reward+=2
                
            if self.agent.start_flag==1:
                collision_reward+=0
                
        return collision_reward
    
    
        
    
    #------------get lidar info
    def _get_lidar_data(self,opp_agent=None):
        """Extract LIDAR distances and hit flags from current observation"""
        a = self.agent
        opp_agent=self.agent2


        sensor_angles = np.linspace(-90, 90, self.n_beams)
        max_range = 300
        
        distances = []
        hits = []
        
        for sa in sensor_angles:
            ray_angle = math.radians(a.angle + sa)
            dx, dy = math.cos(-ray_angle), math.sin(-ray_angle)

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
                # Opponent collision check
                if opp_agent:
                    opponent_rect=opp_agent.rect
                    opponent_mask=opp_agent.mask
                    
                        # Check if the ray point is within the opponent's rect
                    if opponent_rect.collidepoint(x, y):
                            # Convert screen coordinates to opponent's local mask coordinates
                        local_x = x - opponent_rect.x
                        local_y = y - opponent_rect.y
                            
                            # Make sure the local coordinates are within mask bounds
                        if (0 <= local_x < opponent_mask.get_size()[0] and 
                            0 <= local_y < opponent_mask.get_size()[1]):
                            if opponent_mask.get_at((local_x, local_y)) == 1:
                                dist,hit = d,2
                                break
            
            distances.append(dist)
            hits.append(hit)
        
        return distances, hits
    
    #------------get lidar rewards
    def _calculate_lidar_reward(self, distances, hits, steer_action):
        """
        Calculate reward based on LIDAR navigation strategy:
        1. Follow the longest clear (non-hitting) LIDAR line
        2. If all lines are hitting, still follow the longest one
        """
        distances = np.array(distances)
        hits = np.array(hits)
        
        # Sensor angles relative to car heading (-90 to +90 degrees)
        sensor_angles = np.linspace(-90, 90, len(distances))
        
        # Find clear paths (hit = 0)
        clear_mask = (hits == 0)
        car_mask= (hits ==2)
        
        if np.any(clear_mask):
            # Case 1: There are clear paths, follow the longest clear one
            clear_distances = distances.copy()
            clear_distances[~clear_mask] = 0  # Zero out blocked paths
            
            # Find the angle of the longest clear path
            best_clear_idx = np.argmax(clear_distances)
            target_angle = sensor_angles[best_clear_idx]
            
            reward_multiplier = 2.0  # Higher reward for following clear paths
            
        elif np.any (car_mask):
            car_distances=distances.copy()
            car_distances[~car_mask]=0
            nearest_car_idx=np.argmin(car_distances)
            opp_idx= (nearest_car_idx + self.n_beams // 2) % self.n_beams
            search_window = 5  # number of beams on each side to consider
            candidate_indices = [(opp_idx + i) % self.n_beams for i in range(-search_window, search_window+1)]

            # Step 4: Pick the longest clear distance among candidates
            candidate_distances = distances[candidate_indices]
            best_idx_within = candidate_indices[np.argmax(candidate_distances)]

            target_angle = sensor_angles[best_idx_within]
        
            reward_multiplier=2.5 
            
        else:
            # Case 2: All paths are blocked, follow the longest one anyway
            best_blocked_idx = np.argmax(distances)
            target_angle = sensor_angles[best_blocked_idx]
            
            reward_multiplier = 1.5  # Lower reward when all paths blocked
        
        # Calculate steering reward based on how well the action follows the best path
        # Convert target_angle to steering action (-1 to 1)
        # -90° → steer left (-1), 0° → straight (0), +90° → steer right (+1)
        target_steer = (target_angle / 90.0)
        
        
        # Reward for steering towards the best path
        steer_error = abs(steer_action - target_steer)
        steer_reward = (1-steer_error) * reward_multiplier
        
        # Bonus for following longer paths
        max_distance = max(distances)
        distance_bonus = (max_distance / 300.0) * 0.5  # normalized distance bonus
        
        # Additional reward for maintaining distance from walls
        min_side_distance = min(distances[0],distances[1], distances[-2],distances[-1])  # leftmost and rightmost sensors
        wall_clearance_reward = (min_side_distance / 300.0) * 2
        
        total_reward = steer_reward + distance_bonus + wall_clearance_reward
        
        return total_reward


    # -------------------------------
    def _action_to_keys(self, steer, accel):
        keys = {}
        if accel > 0:
            keys[pygame.K_UP] = True
        elif accel < -0.01:
            keys[pygame.K_DOWN] = True
        if steer < -0.01:
            keys[pygame.K_LEFT] = True
        elif steer > 0:
            keys[pygame.K_RIGHT] = True
        return keys

    # -------------------------------
    def render(self):
        if self.render_mode is None:
            return None
            
        # Fill background
        self.screen.fill((0, 150, 0))
        
        
        # Draw track
        self.track.draw(self.screen)
        
        # Draw agent car
        self.agent.draw(self.screen)
        self.agent2.draw(self.screen)
        # Draw checkpoints
        for cp in self.total_cp:
            pygame.draw.circle(self.screen, (0, 0, 255), cp, 10)
        for cp in self.overtaken_cp:
            pygame.draw.circle(self.screen, (0, 255, 0), cp, 10)

        # THIS WAS MISSING - UPDATE THE DISPLAY
        if self.render_mode == "human":
            self.agent.update()
            pygame.display.flip()
            if self.clock:
                self.clock.tick(30)
                
            # Handle events to prevent window freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return False
                    
        return None

    def close(self):
        pygame.quit()

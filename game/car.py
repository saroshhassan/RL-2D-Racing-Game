#car module
import pygame
import math
import numpy as np
import time
from config import SCREEN_HEIGHT,SCREEN_WIDTH, RED
from .utils import RaceTimer
class Car:
    def __init__(self, x, y, width=40, height=70, is_cpu=False, image_path=None, control=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.is_cpu = is_cpu
        self.health = 1000
        self.speed = 0
        self.max_speed = 10
        self.acceleration = 0.2
        self.brake_deceleration = 0.4
        self.free_deceleration = 0.05
        self.angle = 270  # 0 = facing up, positive = clockwise
        self.last_collision_time = 0.0
        self.collision_cooldown = 0.5
        self.timer = RaceTimer()
        self.debug_font = pygame.font.SysFont("Arial", 20)  
        self.boundary_collision_flag=False
        self.finish_dist=0
        self.controls=control
        self.distance=0
        self.start_flag=0
        
        if image_path:
            self.base_image = pygame.image.load(image_path).convert_alpha()
            self.base_image = pygame.transform.scale(self.base_image, (self.width, self.height))
        else:
            raise ValueError("You must provide an image_path for the car sprite.")

        self.image = pygame.image.load(image_path).convert_alpha()
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect(center=(self.x, self.y))
    
    # ---------- Movement ----------
    def accelerate(self):
        self.speed += self.acceleration
        if self.speed > self.max_speed:
            self.speed = self.max_speed
    

    def brake(self):
        self.speed -= self.brake_deceleration
        if self.speed < -self.max_speed / 2:
            self.speed = -self.max_speed / 2

    def coast(self):
        if self.speed > 0:
            self.speed -= self.free_deceleration
            if self.speed < 0:
                self.speed = 0
                
            """    for reverse coasting-reverse and speed becomes zero"""
        elif self.speed < 0:
            self.speed += self.free_deceleration
            if self.speed > 0:
                self.speed = 0

    def turn_left(self):
        if self.speed != 0:
            self.angle += 4

    def turn_right(self):
        if self.speed != 0:
            self.angle -= 4

    def update(self):
        # Convert angle to radians
        rad = math.radians(self.angle)
        dx = -self.speed * math.sin(rad)
        dy = -self.speed * math.cos(rad)

        # Update car position
        self.x += dx
        self.y += dy
        self.rect.center = (self.x, self.y)

        # Rotate car image and update rect
        self.image = pygame.transform.rotate(self.base_image, self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)

        # Update mask for pixel-perfect collisions
        self.mask = pygame.mask.from_surface(self.image)
        self.angle=self.angle%360
        
     #---------------------debugging functions
    def draw_mask(self, screen, color=RED):
        """Convert a mask to a surface and draw it for debugging"""
        mask_surf = self.mask.to_surface(setcolor=color, unsetcolor=(0, 0, 0, 0))
        screen.blit(mask_surf, (self.rect.left,self.rect.top))
        
    #----------------------heuristic retrieval function
    def calculate(self , point): #x=400,y=25, color=(123,123,123)):
        "Calculate euclidean distance heuristic for car"
        finx,finy=point
        
        distance=math.sqrt(((finx-self.rect.centerx)**2)+(finy-self.rect.centery)**2)
        #text=f"Manhattan Distance to finish: {distance}"
        #font =pygame.font.SysFont("Arial", 30)
        #distance_text=font.render(text,True, color)
        #screen.blit(distance_text, (x, y))
        self.update()
        return distance
    
        
    
    #-----------------------movement
    def move(self, keys):
        
        if self.controls=="arrows":
                
            if keys[pygame.K_UP]:
                self.accelerate()
            elif keys[pygame.K_DOWN]:
                self.brake()
            else:
                self.coast()

            if keys[pygame.K_LEFT]:
                self.turn_left()
            if keys[pygame.K_RIGHT]:
                self.turn_right()

            self.update()
            
        elif self.controls =="wasd":
            if keys[pygame.K_w]:
                self.accelerate()
            elif keys[pygame.K_s]:
                self.brake()
            else:
                self.coast()

            if keys[pygame.K_a]:
                self.turn_left()
            if keys[pygame.K_d]:
                self.turn_right()
                
            self.update()
            
            # Clamp car inside screen
        screen_width, screen_height = SCREEN_WIDTH,SCREEN_HEIGHT
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > screen_width:
            self.rect.right = screen_width
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > screen_height:
            self.rect.bottom = screen_height
            
    def cpu_move(self, obs, model):
        """
    obs: observation from environment
    model: trained PPO model
        """
         # Flatten to ensure shape is always (2,)
        
        action,_ = model.predict(obs,deterministic=True)
        action = np.array(action).flatten()
        steer, accel = float(action[0]), float(action[1])

        # --- Map continuous values to controls ---
        # Acceleration / braking
        if accel > 0.1:
            self.accelerate()
        elif accel < -0.1:
            self.brake()
        else:
            self.coast()

        # Steering
        if steer < -0.1:
            self.turn_left()
        elif steer > 0.1:
            self.turn_right()

        # Update car physics
        self.update()

            
            
            
        
            # Clamp car inside screen
        screen_width, screen_height = SCREEN_WIDTH,SCREEN_HEIGHT
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > screen_width:
            self.rect.right = screen_width
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > screen_height:
            self.rect.bottom = screen_height

    # ---------- Rendering ----------
    def draw(self, screen):
        screen.blit(self.image, self.rect)
        
    def show_position(self, surface):
        coords_text = self.debug_font.render(
        f"({int(self.rect.centerx)}, {int(self.rect.centery)})",
        True,
        (12, 12, 12)
        )
        surface.blit(coords_text, (300, 300))
        
    def draw_sensor_rays(self, screen, boundary_mask=None, opponent_mask=None, opponent_rect=None, screen_width=None, screen_height=None):
        """Draw sensor rays for debugging purposes with boundary and opponent collision detection"""
        # Relative sensor angles (relative to car heading)
        sensor_angles = [-90, -60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60, 90]  
        max_range = 300  
        
        # Shift car angle so that 270° (right) becomes 0° forward
        heading = (self.angle - 270) % 360
        
        for sa in sensor_angles:
            # Absolute ray angle in radians
            ray_angle = math.radians((heading) - sa)  # -90 to align sprite starting at 270°
            dx = math.cos(-ray_angle)
            dy = math.sin(-ray_angle)
            dist = max_range
            
            if boundary_mask and screen_width and screen_height:
                for d in range(1, max_range):  # fine step = pixel precision
                    x = int(self.rect.centerx + dx * d)
                    y = int(self.rect.centery + dy * d)
                    
                    # Out of bounds check
                    if not (0 <= x < screen_width and 0 <= y < screen_height):
                        dist = d
                        break
                    
                    # Boundary collision check
                    if boundary_mask.get_at((x, y)) == 1:  # black = wall
                        dist = d
                        break
                    
                    # Opponent collision check
                    if opponent_mask and opponent_rect:
                        # Check if the ray point is within the opponent's rect
                        if opponent_rect.collidepoint(x, y):
                            # Convert screen coordinates to opponent's local mask coordinates
                            local_x = x - opponent_rect.x
                            local_y = y - opponent_rect.y
                            
                            # Make sure the local coordinates are within mask bounds
                            if (0 <= local_x < opponent_mask.get_size()[0] and 
                                0 <= local_y < opponent_mask.get_size()[1]):
                                if opponent_mask.get_at((local_x, local_y)) == 1:
                                    dist = d
                                    break
            
            # End point of ray
            end_x = self.rect.centerx + dx * dist
            end_y = self.rect.centery + dy * dist
            
            # Draw the ray (green)
            pygame.draw.line(screen, (0, 255, 0),
                            (int(self.rect.centerx), int(self.rect.centery)),
                            (int(end_x), int(end_y)), 2)
            
            # Draw collision point (red dot)
            pygame.draw.circle(screen, (255, 0, 0),
                            (int(end_x), int(end_y)), 3)

    
    
    # ---------- Collision / Damage ----------
    def collide(self):
        self.health-=5
        self.boundary_collision_flag=True
    
        
    def take_damage(self, amount=10):
        now = time.time()
        # Only apply damage if enough time passed since last collision
        if now - self.last_collision_time > self.collision_cooldown:
            self.health = max(0, self.health - amount)
            self.last_collision_time = now
            
            
    #------------debugging
    def draw_angle(self, screen, font=None):
        """Draw the car's current angle at its center"""
        if font is None:
            font = pygame.font.SysFont("Arial", 20)

        # Render angle as text
        angle_text = font.render(str(int(self.angle)), True, (255, 255, 0))  # yellow text
        text_rect = angle_text.get_rect(center=self.rect.center)

        # Blit onto the screen
        screen.blit(angle_text, text_rect)

    

            

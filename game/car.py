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
        self.health = 100
        self.speed = 0
        self.max_speed = 10
        self.acceleration = 0.2
        self.brake_deceleration = 0.4
        self.free_deceleration = 0.05
        self.angle = 270  # 0 = facing up, positive = clockwise
        self.last_collision_time = 0.0
        self.collision_cooldown = 0.1
        self.timer = RaceTimer()
        self.debug_font = pygame.font.SysFont("Arial", 20)  
        self.boundary_collision_flag=False
        self.finish_dist=0
        self.controls=control
        self.distance=0
        
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
        
     #---------------------debugging functions
    def draw_mask(self, screen, color=RED):
        """Convert a mask to a surface and draw it for debugging"""
        mask_surf = self.mask.to_surface(setcolor=color, unsetcolor=(0, 0, 0, 0))
        screen.blit(mask_surf, (self.rect.left,self.rect.top))
        
    #----------------------heuristic retrieval function
    def calculate(self ,screen): #x=400,y=25, color=(123,123,123)):
        "Calculate Manhattan distance heuristic for car"
        finx,finy=(1042,150)
        
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
            
   

            

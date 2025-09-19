#utils.py-game utilities
import pygame
import time
import os
from config import RED


def draw_health_bar(screen, x, y, health, max_health=100, width=100, height=10):
    ratio = health / max_health
    pygame.draw.rect(screen, (255,0,0), (x, y, width, height))  # background
    pygame.draw.rect(screen, (0,255,0), (x, y, width * ratio, height))  # health


def log_lap_time(player_name, lap_time, filename="logs/times.txt"):
    os.makedirs("logs", exist_ok=True)  # ensure logs directory exists
    with open(filename, "a") as f:
        f.write(f"{player_name}: {lap_time:.2f} seconds\n")
        
"""def log_lap_actions(player_name, lap_time, filename="dataset/v1.txt"):
    os.makedirs("dataset",exist_ok=True)
    with open(filename, "a") as f:
        f.write(f{"player_name"}:  )"""

class RaceTimer:
    def __init__(self, font=None):
        self.start_time = None
        self.end_time = None
        self.running = False
        self.final_time = None

        # Use passed font or make a default one
        self.font = font or pygame.font.SysFont("Arial", 30)

    def start(self):
        """Start or restart the timer."""
        self.start_time = time.time()
        self.end_time = None
        self.running = True
        self.final_time = None
        
    def reset (self):
        "Reset timer to OG settings"
        self.start_time = None
        self.end_time = None
        self.running = False
        self.final_time = None
            
    def stop(self):
        """Stop the timer and freeze final time."""
        if self.running:
            self.end_time = time.time()
            self.running = False
            self.final_time = self.end_time - self.start_time

    def get_time(self):
        """Return elapsed time in seconds (running or final)."""
        if self.running:
            return time.time() - self.start_time
        elif self.final_time is not None:
            return self.final_time
        else:
            return 0.0

    def draw(self, screen, x=350, y=20, color=(0, 0, 0)):
        """Draw timer on screen (topleft by default)."""
        elapsed = self.get_time()
        text = f"Time: {elapsed:.2f}s"
        label = self.font.render(text, True, color)
        screen.blit(label, (x, y))
 
def _action_to_keys(steer, accel):
        """Convert continuous actions to discrete key presses"""
        keys = {}
        #.
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

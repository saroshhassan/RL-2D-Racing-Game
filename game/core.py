#core.py- core game functions
import pygame, sys
import numpy as np
from config import SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, RED, BLUE, BLACK
from .car import Car
from .track import Track
from .collision import check_collision, check_boundaries, check_boundary_collision
from .utils import draw_health_bar, RaceTimer, log_lap_time, _action_to_keys
from .game_over import draw_game_over
from .mouse_debug import draw_mouse_debug, toggle_mouse_debug
from stable_baselines3 import PPO
from .observations import make_obs



class Game:
    def __init__(self,twoPlayer=False,vs_cpu=False):
        self.vs_cpu = vs_cpu
        self.twoPlayer=twoPlayer
        self.state="playing"
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Car Race")

        self.clock = pygame.time.Clock()
        self.running = True

        # Player car
        self.player = Car(90, 625,image_path= "assets/car/car1.png", is_cpu=False, control="arrows") 
        
        #Player 2 car
        self.player2 = Car(90, 715,image_path="assets/car/car2.png", is_cpu=False, control= "wasd") if twoPlayer else None

        # CPU car
        self.cpu = Car(90, 715,image_path="assets/car/car2.png", is_cpu=True, control= None) if vs_cpu else None
        
        # Load final trained model
        self.cpu_model = PPO.load("models/pyrace_cpu_ppo") if vs_cpu else None


        self.track = Track()
        
      
            
    def run(self):
        while self.running:
            keys = pygame.key.get_pressed()

            # --- Event handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_m:  # Press M to toggle mouse debug
                        toggle_mouse_debug()

                if self.state == "game_over":
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:  # restart
                            self.__init__(vs_cpu=self.vs_cpu, twoPlayer=self.twoPlayer)
                            return 'restart'
                        elif event.key == pygame.K_q:  # quit
                            return 'menu'

            # --- Drawing ---
            self.screen.fill(WHITE)
            self.track.draw(self.screen)
            draw_mouse_debug(self.screen)
           
            # Always draw cars
            self.player.draw(self.screen)
            #self.player.display_dist(self.screen)
            #self.player.draw_mask(self.screen)
            self.player.draw_angle(self.screen)
            if self.twoPlayer:
                self.player2.draw(self.screen)
                self.player.draw_sensor_rays(screen=self.screen,boundary_mask=self.track.boundary_mask, opponent_mask=self.player2.mask, opponent_rect=self.player2.rect, screen_width=SCREEN_WIDTH,screen_height=SCREEN_HEIGHT)
            
                
            if self.cpu:
                self.cpu.draw(self.screen)
                self.player.draw_sensor_rays(screen=self.screen,boundary_mask=self.track.boundary_mask, opponent_mask=self.cpu.mask, opponent_rect=self.cpu.rect, screen_width=SCREEN_WIDTH,screen_height=SCREEN_HEIGHT)
            
                

            # Always draw HUD
            draw_health_bar(self.screen, 50, 20, self.player.health)
            self.player.timer.draw(self.screen, 50, 50, BLACK)
            
            if self.cpu:
                draw_health_bar(self.screen, 600, 20, self.cpu.health)
                self.cpu.timer.draw(self.screen, 600, 50, BLACK)
                
            if self.twoPlayer:
                draw_health_bar(self.screen, 800, 20, self.player2.health)
                self.player2.timer.draw(self.screen, 800, 50 , BLACK)

            if self.state == "playing":
                # --- Gameplay logic ---
                self.player.move(keys)
                
                if self.twoPlayer:
                    self.player2.move(keys)
                
                if self.cpu:
                    # CPU with RL policy
                    obs = make_obs(self.cpu, self.track, SCREEN_WIDTH, SCREEN_HEIGHT)
                    #obs = np.expand_dims(obs, axis=0)  # add batch dimension
                    #action, _ = self.cpu_model.predict(obs, deterministic=True)
                    #action, _ = self.cpu_model.predict(obs, deterministic=True)

                    #action = np.array(action).flatten()

                    #if action.shape[0] != 2:
                    #     raise ValueError(f"Expected action of shape (2,), got {action.shape} with value {action}")

                    #steer, accel = float(action[0]), float(action[1])
                    #^^^correctly unpacked?
                    # convert action into car commands REMINDER UNDERSCORE WAS ADDED AS CHECK
                    #steer, accel = action
                    #cpu_keys = _action_to_keys(steer, accel)
                    self.cpu.cpu_move(obs,self.cpu_model)
                    
                
                if self.player2 and check_collision(self.player,self.player2):
                    self.player.collide()
                    self.player2.collide()
                    
                # Collisions-car car
                if self.cpu and check_collision(self.player, self.cpu):
                    self.player.collide()
                    self.cpu.collide()

                # Track interactions
                if check_boundary_collision(self.player, self.track):
                    self.player.collide()
                if self.player2:
                     if check_boundary_collision(self.player2,self.track):
                         self.player2.collide()
                     
                     
                #Timer start flags    
                status_p1=check_boundaries(self.player,self.track)
                
                #if self.boundary_collision ==True:
                #    self.player.collide()
                if status_p1 == "start":
                    if not self.player.timer.running:
                        self.player.timer.start()
                elif status_p1 == "finish":
                    if self.player.timer.running:
                        self.player.timer.stop()
                        lap_time = self.player.timer.get_time()
                        log_lap_time("Player", lap_time)
                        self.state = "game_over"
                        
                if self.player2:
                    status_p2 = check_boundaries(self.player2, self.track)
                    if check_boundary_collision(self.player2,self.track):
                        self.player2.collide()
                    if status_p2 == "start":
                        if not self.player2.timer.running:
                            self.player2.timer.start()
                    elif status_p2 == "finish":
                        if self.player2.timer.running:
                            self.player2.timer.stop()
                            lap_time = self.player2.timer.get_time()
                            log_lap_time("Player 2", lap_time)
                            self.state = "game_over"

                if self.cpu:
                    status = check_boundaries(self.cpu, self.track)
                    if check_boundary_collision(self.cpu,self.track):
                        self.cpu.collide()
                    if status == "start":
                        if not self.cpu.timer.running:
                            self.cpu.timer.start()
                    elif status == "finish":
                        if self.cpu.timer.running:
                            self.cpu.timer.stop()
                            lap_time = self.cpu.timer.get_time()
                            log_lap_time("CPU", lap_time)
                            self.state = "game_over"

                # Health check
                if self.player.health <= 0 or (self.player2 and self.player2.health <= 0):
                    self.state = "game_over"
                    self.player.timer.stop()
                    if self.player2:
                        self.player2.timer.stop()

                
                if self.player.health <= 0 or (self.cpu and self.cpu.health <= 0):
                    self.state = "game_over"
                    self.player.timer.stop()
                    if self.cpu:
                        self.cpu.timer.stop()

            elif self.state == "game_over":
                # Overlay GAME OVER screen on top of everything
                draw_game_over(self.screen)

            pygame.display.flip()
            self.clock.tick(60)                 


                
    
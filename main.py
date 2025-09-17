# main.py-main module
import pygame
import sys
from config import SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, BLACK
from game.core import Game
from game import mouse_debug

#from env.car_race_env import CarRaceEnv

pygame.init()

FONT = pygame.font.SysFont("Arial", 40)

def draw_text(screen, text, pos, color=BLACK):
    label = FONT.render(text, True, color)
    screen.blit(label, pos)

def main_menu():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Car Racing Game")

    clock = pygame.time.Clock()
    selected = 0
    options = ["Single Player", "1V1", "Vs CPU", "Quit"]

    while True:
        screen.fill(WHITE)

        for i, option in enumerate(options):
            color = (0, 128, 0) if i == selected else BLACK
            draw_text(screen, option, (SCREEN_WIDTH//2 - 100, 200 + i*60), color)

        pygame.display.flip()
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(options)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(options)
                elif event.key == pygame.K_RETURN:
                    if options[selected] == "Single Player":
                        result=Game(twoPlayer=False,vs_cpu=False).run()
                    elif options[selected] == "1V1":
                        result=Game(twoPlayer=True,vs_cpu=False).run()
                    elif options[selected] == "Vs CPU":
                        result=Game(twoPlayer=False, vs_cpu=True).run()
                    elif options[selected] == "Quit":
                        pygame.quit()
                        sys.exit()
                        
                    # Handle results after the game ends
                    if result == "restart":
                        continue  # just re-run the menu loop
                    elif result == "menu":
                        continue  # back to main menu
                        
        

if __name__ == "__main__":
    main_menu()



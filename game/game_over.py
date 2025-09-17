#game_over.py- game over display module
from config import SCREEN_WIDTH, SCREEN_HEIGHT
import pygame

def draw_game_over(screen):
    font = pygame.font.SysFont("Arial", 60, bold=True)
    small_font = pygame.font.SysFont("Arial", 40)

    game_over_text = font.render("GAME OVER", True, (255, 0, 0))
    restart_text = small_font.render("Press R to Restart or Q to Quit", True, (0, 0, 0))

    screen.blit(game_over_text, (SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 - 80))
    screen.blit(restart_text, (SCREEN_WIDTH // 2 - 220, SCREEN_HEIGHT // 2))

    pygame.display.update()
#track.py- TRACK on which CAR drives, not to track movement of car
import pygame
from config import SCREEN_WIDTH, SCREEN_HEIGHT

class Track:
    def __init__(self, image_path="assets/track/TRACKv2.png"):
        # Load and scale track background
        self.image=pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
        self.image = pygame.image.load_extended(image_path).convert()
        self.image = pygame.transform.scale(self.image, (SCREEN_WIDTH, SCREEN_HEIGHT))

        # Track rect
        self.rect = self.image.get_rect(topleft=(0, 0))

        # --- Create masks ---
        # 1. Road (white only, ignore red/green)
            
        self.road_mask = pygame.mask.from_threshold(
            self.image,
            (255, 255, 255, 255),   # pure white
            (30, 30, 30, 255)       # small tolerance
        )

        # Start line (red)
        self.start_mask = pygame.mask.from_threshold(
            self.image,
            (255, 0, 0, 255),
            (30, 30, 30, 255)
        )

        # Finish line (green)
        self.finish_mask = pygame.mask.from_threshold(
            self.image,
            (0, 255, 0, 255),
            (30, 30, 30, 255)
        )
        
        # Boundary mask (purple overlay for debugging)
        self.boundary_mask = pygame.mask.from_threshold(
        self.image,
        (145, 145, 145, 255),   # target boundary color (#919191)
        (20, 20, 20, 255)       # tolerance
        )



        # Extra boundaries (screen edges)
        self.boundary_left = pygame.Rect(0, 0, 40, SCREEN_HEIGHT)
        self.boundary_right = pygame.Rect(SCREEN_WIDTH - 40, 0, 40, SCREEN_HEIGHT)
        
      # --- Utility checks ---
    def is_on_road(self, x, y):
        """True if position is drivable road (white only)"""
        return self.road_mask.get_at((int(x), int(y))) == 1

    def is_on_start(self, x, y):
        """True if position overlaps start line (red)"""
        return self.start_mask.get_at((int(x), int(y))) == 1

    def is_on_finish(self, x, y):
        """True if position overlaps finish line (green)"""
        return self.finish_mask.get_at((int(x), int(y))) == 1

    def draw(self, screen):
        screen.blit(self.image, (0, 0))
        
    def get_pixel_color(self, x, y):
        """Returns (R, G, B) at the given position"""
        if 0 <= x < self.image.get_width() and 0 <= y < self.image.get_height():
            return self.image.get_at((x, y))[:3]  # ignore alpha
        return (0, 0, 0)  # treat out-of-bounds as barrier
    #----------debug code
    def draw_rect(self, screen, rect=None, color=(0, 0, 255), width=2):
        
        """Draws the track's bounding rectangle for debugging.
        Args:
            screen: pygame display surface
            rect: pygame.Rect (defaults to track.rect)
            color: RGB tuple for outline
        width: thickness of the outline
        """
        if rect is None:
            rect = self.rect
        pygame.draw.rect(screen, color, rect, width)
        
    def draw_debug_masks(self, screen):
        """Overlay road (blue), start (red), and finish (green) masks on track."""
        # Road = blue
        road_surf = self.road_mask.to_surface(setcolor=(0, 0, 255, 120),
                                            unsetcolor=(0, 0, 0, 0))
        # Start = red
        start_surf = self.start_mask.to_surface(setcolor=(255, 0, 0, 150),
                                                unsetcolor=(0, 0, 0, 0))
        # Finish = green
        finish_surf = self.finish_mask.to_surface(setcolor=(0, 255, 0, 150),
                                                unsetcolor=(0, 0, 0, 0))
        
        # Boundary = purple
        boundary_surf = self.boundary_mask.to_surface(setcolor=(128, 0, 128, 150),
                                              unsetcolor=(0, 0, 0, 0))

        # Blit overlays
        screen.blit(road_surf, (0, 0))
        screen.blit(start_surf, (0, 0))
        screen.blit(finish_surf, (0, 0))
        screen.blit(boundary_surf, (0, 0))



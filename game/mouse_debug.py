import pygame

# Debug toggle flag
SHOW_MOUSE_DEBUG = False

# Font (initialized later after pygame.init())
_font = None

def init_mouse_debug():
    """Initialize font once (call after pygame.init())."""
    global _font
    if _font is None:
        _font = pygame.font.SysFont("Arial", 20)

def toggle_mouse_debug():
    """Toggle mouse debug overlay on/off."""
    global SHOW_MOUSE_DEBUG
    SHOW_MOUSE_DEBUG = not SHOW_MOUSE_DEBUG

def draw_mouse_debug(screen):
    """Draw red dot at mouse position and coords (if enabled)."""
    if not SHOW_MOUSE_DEBUG:
        return

    global _font
    if _font is None:
        init_mouse_debug()

    # Mouse position
    mouse_x, mouse_y = pygame.mouse.get_pos()

    # Red dot
    pygame.draw.circle(screen, (255, 0, 0), (mouse_x, mouse_y), 5)

    # Render coords text
    coords_text = _font.render(f"({mouse_x}, {mouse_y})", True, (0, 0, 0))
    text_rect = coords_text.get_rect(topright=(screen.get_width() - 10, 10))
    screen.blit(coords_text, text_rect)

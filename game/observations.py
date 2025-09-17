import math
import numpy as np
#import car
from config import SCREEN_HEIGHT, SCREEN_WIDTH

def make_obs(agent, opponent, width=SCREEN_WIDTH, height=SCREEN_HEIGHT, max_speed=8):
    """
    Build observation vector [ax, ay, sinθa, cosθa, va,
                               ox, oy, sinθo, cosθo, vo]
    from two Car objects.
    """
    return np.array([
        (agent.rect.centerx / width - 0.5) * 2,
        (agent.rect.centery / height - 0.5) * 2,
        math.sin(math.radians(agent.angle)),
        math.cos(math.radians(agent.angle)),
        agent.speed / max_speed,
        (opponent.rect.centerx / width - 0.5) * 2,
        (opponent.rect.centery / height - 0.5) * 2,
        math.sin(math.radians(opponent.angle)),
        math.cos(math.radians(opponent.angle)),
        opponent.speed / max_speed
    ], dtype=np.float32)

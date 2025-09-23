import math
import numpy as np
from config import SCREEN_HEIGHT, SCREEN_WIDTH

def make_obs(agent, track, width=SCREEN_WIDTH, height=SCREEN_HEIGHT, max_speed=8):
    """
    Build observation vector [ax, ay, sinθa, cosθa, va, time,
                               lidar_distances, lidar_hits]
    from Car object.
    """
    a = agent
    dist=a.last_dist_to_target
    if dist==None:
        obs_dist=0
    else:
        obs_dist=dist    
    n_beams = 13
    obs = [
        (a.rect.centerx / width - 0.5) * 2,
        (a.rect.centery / height - 0.5) * 2,
        math.sin(math.radians(a.angle)),
        math.cos(math.radians(a.angle)),
        a.speed / max_speed,
        a.timer.get_time() / 100.0,  # normalize time
        obs_dist/1000.0,
    ]

    # LIDAR beams
    sensor_angles = np.linspace(-90, 90, n_beams)  # relative to heading
    max_range = 300

    for sa in sensor_angles:
        ray_angle = math.radians(a.angle + sa)
        dx, dy = math.cos(-ray_angle), math.sin(-ray_angle)

        dist = max_range
        hit = 0
        
        for d in range(1, max_range, 3):
            x = int(a.rect.centerx + dx * d)
            y = int(a.rect.centery + dy * d)

            if not (0 <= x < width and 0 <= y < height):
                dist, hit = d, 1
                break
            if track.boundary_mask.get_at((x, y)) == 1:
                dist, hit = d, 1
                break
        
        # Append results OUTSIDE the distance loop
        obs.append(dist / max_range)  # normalized distance
        obs.append(hit)               # 1 = collision boundary, 0 = free
            
    obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)     
    return np.array(obs, dtype=np.float32)
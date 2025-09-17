#collision.py- collision mechanics
import pygame
import math


import pygame

def check_collision(player, other):
    """
    Mask-based collision check between two Car objects.
    Returns True if masks overlap, else False.
    """
    if not player or not other:
        return False

    # Offset = other relative to player
    offset = (other.rect.left - player.rect.left,
              other.rect.top - player.rect.top)

    overlap = player.mask.overlap(other.mask, offset)

    if overlap:  # collision point found
        #player.collide()
        #other.collide()
        return True

    return False



def check_boundary_collision(car, track):
    offset = (car.rect.left - track.rect.left, car.rect.top - track.rect.top)
    overlap = track.boundary_mask.overlap(car.mask, offset)

    if overlap:  # collision point found
        return True
    return False

def check_boundaries(car, track):
    """
    Returns:
        "road"   -> safe driving
        "barrier"-> hit wall / off-road
        "start"  -> crossed start line
        "finish" -> crossed finish line
    """

    # offset = position of car relative to track
    offset = (car.rect.left , car.rect.top )

    # --- Start line ---
    if track.start_mask.overlap(car.mask, offset):
        return "start"

    # --- Finish line ---
    if track.finish_mask.overlap(car.mask, offset):
        return "finish"

    # --- Road ---
    if track.road_mask.overlap(car.mask, offset):
        return "road"

    # --- Else it's a barrier ---
    return "barrier"

def resolve_overlap(mover, other, track, max_steps=20):
    # Try to step mover back along its velocity until overlap clears
    offset = (other.rect.left - mover.rect.left, other.rect.top - mover.rect.top)
    overlap = mover.mask.overlap(other.mask, offset)
    if not overlap:
        return False
    rad = math.radians(mover.angle)
    dx = -mover.speed * math.sin(rad) * 0.5
    dy = -mover.speed * math.cos(rad) * 0.5
    steps = 0
    while overlap and steps < max_steps:
        mover.rect.x += int(dx)
        mover.rect.y += int(dy)
        mover.update_image_and_mask()
        offset = (other.rect.left - mover.rect.left, other.rect.top - mover.rect.top)
        overlap = mover.mask.overlap(other.mask, offset)
        steps += 1
    return not bool(overlap)


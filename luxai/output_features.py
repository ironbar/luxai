"""
Output features for imitation learning
"""
import numpy as np


def create_actions_mask(active_units_to_position, observation):
    width, height = observation['width'], observation['height']
    mask = np.zeros((width, height, 1), dtype=np.float32)
    for position in active_units_to_position.values():
        x, y = position
        mask[x, y] = 1
    return mask
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

UNIT_ACTIONS_MAP = {

}


CITY_ACTIONS_MAP = {
    'r': 0, # research
    'bw': 1, # build worker
    'bc': 2, # build cart
}


def create_output_features(actions, units_to_position, observation):
    width, height = observation['width'], observation['height']

    city_actions = np.zeros((len(CITY_ACTIONS_MAP), width, height), dtype=np.float32)
    for action in actions:
        splits = action.split(' ')
        action_id = splits[0]
        if action_id in CITY_ACTIONS_MAP:
            x, y = int(splits[1]), int(splits[2])
            city_actions[CITY_ACTIONS_MAP[action_id], x, y] = 1
    return city_actions
"""
Output features for imitation learning
"""
import warnings
import numpy as np


def create_actions_mask(active_unit_to_position, observation):
    width, height = observation['width'], observation['height']
    mask = np.zeros((width, height, 1), dtype=np.float32)
    for position in active_unit_to_position.values():
        x, y = position
        mask[x, y] = 1
    return mask


UNIT_ACTIONS_MAP = {
    'm n': 0, # move north
    'm e': 1, # move east
    'm s': 2, # move south
    'm w': 3, # move west
    't n': 4, # transfer north
    't e': 5, # transfer east
    't s': 6, # transfer south
    't w': 7, # transfer west
    'bcity': 8, # build city
    'p': 9, # pillage
}


CITY_ACTIONS_MAP = {
    'r': 0, # research
    'bw': 1, # build worker
    'bc': 2, # build cart
}


def create_output_features(actions, unit_to_position, observation):
    width, height = observation['width'], observation['height']

    unit_actions = np.zeros((len(UNIT_ACTIONS_MAP), width, height), dtype=np.float32)
    city_actions = np.zeros((len(CITY_ACTIONS_MAP), width, height), dtype=np.float32)
    for action in actions:
        splits = action.split(' ')
        action_id = splits[0]
        if action_id in CITY_ACTIONS_MAP:
            x, y = int(splits[1]), int(splits[2])
            city_actions[CITY_ACTIONS_MAP[action_id], x, y] = 1
        elif action_id == 'm': # move
            unit_id, direction = splits[1], splits[2]
            x, y = unit_to_position[unit_id]
            if direction == 'c':
                continue
            unit_actions[UNIT_ACTIONS_MAP['%s %s' % (action_id, direction)], x, y] = 1
        elif action_id == 't': # transfer
            unit_id, dst_id = splits[1], splits[2]
            try:
                x, y = unit_to_position[unit_id]
                x_dst, y_dst = unit_to_position[dst_id]
                direction = get_transfer_direction(x, y, x_dst, y_dst)
                unit_actions[UNIT_ACTIONS_MAP['%s %s' % (action_id, direction)], x, y] = 1
            except KeyError:
                # I have found that for 26458198.json player 0 there is an incorrect transfer action
                warnings.warn('Could not create transfer action because there were missing units')
            except SamePositionException:
                warnings.warn('Could not create transfer action because source and dst unit are at the same place')
        elif action_id in {'bcity', 'p'}:
            unit_id = splits[1]
            x, y = unit_to_position[unit_id]
            unit_actions[UNIT_ACTIONS_MAP[action_id], x, y] = 1
    # to channels last convention
    unit_actions = np.transpose(unit_actions, axes=(1, 2, 0))
    city_actions = np.transpose(city_actions, axes=(1, 2, 0))
    return unit_actions, city_actions

class SamePositionException(Exception):
    pass

def get_transfer_direction(x_source, y_source, x_dst, y_dst):
    if x_dst < x_source:
        return 'w'
    elif x_dst > x_source:
        return 'e'
    elif y_dst < y_source:
        return 'n'
    elif y_dst > y_source:
        return 's'
    else:
        raise SamePositionException('Could not compute transfer direction for: %s' % str((x_source, y_source, x_dst, y_dst)))
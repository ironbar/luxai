"""
Functions for generating actions from model predictions
"""
import numpy as np

from luxai.input_features import parse_unit_info
from luxai.output_features import CITY_ACTIONS_MAP, UNIT_ACTIONS_MAP


def create_actions_for_cities_from_model_predictions(preds, active_cities_to_position, action_threshold=0.5):
    """
    Creates actions in the luxai format from the predictions of the model

    Parameters
    ----------
    preds : 3d array
        3d array with the predictions being the dimensions (x, y, action)
    active_cities_to_position : dict
        A dictionary that maps city tile identifier to x, y position
    action_threshold : float
        A threshold that filters predictions with a smaller value than that
    """
    actions = []
    idx_to_action = {idx: name for name, idx in CITY_ACTIONS_MAP.items()}
    for position in active_cities_to_position.values():
        x, y = position
        city_preds = preds[x, y]
        action_idx = np.argmax(city_preds)
        if city_preds[action_idx] > action_threshold:
            action_key = idx_to_action[action_idx]
            actions.append('%s %i %i' % (action_key, x, y))
    return actions


def create_actions_for_units_from_model_predictions(
        preds, active_units_to_position, units_to_position, observation, action_threshold=0.5):
    """
    Creates actions in the luxai format from the predictions of the model

    Parameters
    ----------
    preds : 3d array
        3d array with the predictions being the dimensions (x, y, action)
    active_units_to_position : dict
        A dictionary that maps active unit identifier to x,y position
    units_to_position : dict
        A dictionary that maps all unit identifier to x,y position
    action_threshold : float
        A threshold that filters predictions with a smaller value than that
    """
    preds = preds.copy()
    actions = []
    idx_to_action = {idx: name for name, idx in UNIT_ACTIONS_MAP.items()}
    for unit_id, position in active_units_to_position.items():
        x, y = position
        unit_preds = preds[x, y]
        action_idx = np.argmax(unit_preds)
        if unit_preds[action_idx] > action_threshold:
            action_key = idx_to_action[action_idx]
            actions.append(create_unit_action(action_key, unit_id, units_to_position, observation))
            # This ensures that units with overlap do not repeat actions
            preds[x, y, action_idx] = 0
    # TODO: deal with collisions
    return actions


def create_unit_action(action_key, unit_id, units_to_position, observation):
    action_id = action_key.split(' ')[0]
    if action_id == 'm':
        action = 'm %s %s' % (unit_id, action_key.split(' ')[-1])
        return action
    elif action_id in ['bcity', 'p']:
        action = '%s %s' % (action_id, unit_id)
        return action
    elif action_id == 't':
        direction = action_key.split(' ')[1]
        position = units_to_position[unit_id]
        dst_position = _get_dst_position(position, direction)
        dst_unit_id = _find_unit_in_position(dst_position, units_to_position)
        resource, amount = _get_most_abundant_resource_from_unit(unit_id, observation)
        return 't %s %s %s %i' % (unit_id, dst_unit_id, resource, amount)
    else:
        raise KeyError(action_id)


def _get_dst_position(position, direction):
    if direction == 'n':
        dst_position = (position[0], position[1] - 1)
    elif direction == 'e':
        dst_position = (position[0] + 1, position[1])
    elif direction == 's':
        dst_position = (position[0], position[1] + 1)
    elif direction == 'w':
        dst_position = (position[0] - 1, position[1])
    return dst_position


def _find_unit_in_position(position, units_to_position):
    for other_unit_id, other_position in units_to_position.items():
        if other_position == position:
            return other_unit_id


def _get_most_abundant_resource_from_unit(unit_id, observation):
    key = ' %s ' % unit_id
    for update in observation['updates']:
        if key in update:
            resources = parse_unit_info(update.split(' '))[-3:]
            resource_names = ['wood', 'coal', 'uranium']
            idx = np.argmax(resources)
            return resource_names[idx], resources[idx]
    raise KeyError(unit_id)


def remove_collision_actions(actions, unit_to_position, unit_to_action, city_positions):
    blocked_positions = get_blocked_positions_using_units_that_do_not_move(unit_to_position, unit_to_action, city_positions)


def get_blocked_positions_using_units_that_do_not_move(unit_to_position, unit_to_action, city_positions):
    """
    Returns a set of positions of units that do not move and are outside a city
    """
    blocked_positions = set()
    for unit_id, position in unit_to_position.items():
        if unit_id in unit_to_action:
            action = unit_to_action[unit_id]
            if not action.startswith('m ') and position not in city_positions:
                blocked_positions.add(position)
        else:
            if position not in city_positions:
                blocked_positions.add(position)
    return blocked_positions

"""
Functions for generating actions from model predictions
"""
import numpy as np
import random

from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game_constants import GAME_CONSTANTS

from luxai.input_features import parse_unit_info
from luxai.output_features import CITY_ACTIONS_MAP, UNIT_ACTIONS_MAP


def create_actions_for_cities_from_model_predictions(preds, active_city_to_position,
                                                     empty_unit_slots, action_threshold=0.5,
                                                     is_post_processing_enabled=True,
                                                     policy='greedy'):
    """
    Creates actions in the luxai format from the predictions of the model

    Parameters
    ----------
    preds : 3d array
        3d array with the predictions being the dimensions (x, y, action)
    active_city_to_position : dict
        A dictionary that maps city tile identifier to x, y position
    action_threshold : float
        A threshold that filters predictions with a smaller value than that
    empty_unit_slots : int
        Number of units that can be build
    is_post_processing_enabled: bool
        If true it won't build units if there are not empty_unit_slots
    policy : str
        Name of the policy we want to use to choose action, f.e. greedy
    """
    preds = preds.copy()
    actions = []
    idx_to_action = {idx: name for name, idx in CITY_ACTIONS_MAP.items()}
    city_to_priority = {city_id: np.max(preds[x, y]) for city_id, (x, y) in active_city_to_position.items()}
    for city_id in rank_units_based_on_priority(city_to_priority):
        x, y = active_city_to_position[city_id]
        city_preds = preds[x, y]
        if empty_unit_slots <= 0 and is_post_processing_enabled:
            city_preds[CITY_ACTIONS_MAP['bw']] = 0
            city_preds[CITY_ACTIONS_MAP['bc']] = 0
        action_idx = choose_action_idx_from_predictions(city_preds, policy, action_threshold)
        if action_idx is not None:
            action_key = idx_to_action[action_idx]
            actions.append('%s %i %i' % (action_key, x, y))
            if action_key in ['bw', 'bc']:
                empty_unit_slots -= 1
    return actions


def choose_action_idx_from_predictions(preds, policy, action_threshold):
    if policy == 'greedy':
        action_idx = np.argmax(preds)
        if preds[action_idx] <= action_threshold:
            action_idx = None
    elif policy == 'random':
        candidate_indices = [idx for idx, pred in enumerate(preds) if pred > action_threshold]
        if candidate_indices:
            action_idx = random.choices(candidate_indices,
                                        weights=[preds[idx] for idx in candidate_indices])[0]
        else:
            action_idx = None
    return action_idx


def create_actions_for_units_from_model_predictions(
        preds, active_unit_to_position, unit_to_position, observation, city_positions,
        action_threshold=0.5, is_post_processing_enabled=True, policy='greedy'):
    """
    Creates actions in the luxai format from the predictions of the model

    Parameters
    ----------
    preds : 3d array
        3d array with the predictions being the dimensions (x, y, action)
    active_unit_to_position : dict
        A dictionary that maps active unit identifier to x,y position
    unit_to_position : dict
        A dictionary that maps all unit identifier to x,y position
    observation : dict
        Dictionary with the observation of the game
    city_positions : set or list
        A set with all the positions of the cities
    action_threshold : float
        A threshold that filters predictions with a smaller value than that
    is_post_processing_enabled: bool
        If true actions with collisions will be removed and cities won't be built if not enough
        resources are available
    """
    preds = preds.copy()
    if is_post_processing_enabled:
        preds = apply_can_city_be_built_mask_to_preds(preds, active_unit_to_position, observation)
    idx_to_action = {idx: name for name, idx in UNIT_ACTIONS_MAP.items()}
    unit_to_action, unit_to_priority = {}, {}
    for unit_id, position in active_unit_to_position.items():
        x, y = position
        unit_preds = preds[x, y]
        action_idx = choose_action_idx_from_predictions(unit_preds, policy, action_threshold)
        if action_idx is not None:
            action_key = idx_to_action[action_idx]
            unit_to_action[unit_id] = create_unit_action(action_key, unit_id, unit_to_position, observation)
            unit_to_priority[unit_id] = unit_preds[action_idx]
            # This ensures that units with overlap do not repeat actions
            preds[x, y, action_idx] = 0
    if is_post_processing_enabled:
        remove_collision_actions(unit_to_action, unit_to_position, unit_to_priority, city_positions)
    return list(unit_to_action.values())


def create_unit_action(action_key, unit_id, unit_to_position, observation):
    action_id = action_key.split(' ')[0]
    if action_id == 'm':
        action = 'm %s %s' % (unit_id, action_key.split(' ')[-1])
        return action
    elif action_id in ['bcity', 'p']:
        action = '%s %s' % (action_id, unit_id)
        return action
    elif action_id == 't':
        direction = action_key.split(' ')[1]
        position = unit_to_position[unit_id]
        dst_position = _get_dst_position(position, direction)
        dst_unit_id = _find_unit_in_position(dst_position, unit_to_position)
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
    elif direction == 'c':
        dst_position = position
    else:
        raise KeyError(direction)
    return dst_position


def _find_unit_in_position(position, unit_to_position):
    for other_unit_id, other_position in unit_to_position.items():
        if other_position == position:
            return other_unit_id


def _get_most_abundant_resource_from_unit(unit_id, observation):
    resources = _get_unit_resources(unit_id, observation)
    resource_names = ['wood', 'coal', 'uranium']
    idx = np.argmax(resources)
    return resource_names[idx], resources[idx]


def _get_unit_resources(unit_id, observation):
    key = ' %s ' % unit_id
    for update in observation['updates']:
        if key in update:
            resources = parse_unit_info(update.split(' '))[-3:]
            return resources
    raise KeyError(unit_id)


def remove_collision_actions(unit_to_action, unit_to_position, unit_to_priority, city_positions):
    blocked_positions = get_blocked_positions_using_units_that_do_not_move(
        unit_to_position, unit_to_action, city_positions)
    for unit_id in rank_units_based_on_priority(unit_to_priority):
        action = unit_to_action[unit_id]
        if action.startswith('m '):
            direction = action.split(' ')[-1]
            position = unit_to_position[unit_id]
            next_position = _get_dst_position(position, direction)
            if next_position in blocked_positions:
                unit_to_action.pop(unit_id)
            elif next_position not in city_positions:
                blocked_positions.add(next_position)


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


def rank_units_based_on_priority(unit_to_priority):
    units = np.array(list(unit_to_priority.keys()))
    priority = [unit_to_priority[unit_id] for unit_id in units]
    return units[np.argsort(priority)[::-1]].tolist()


def apply_can_city_be_built_mask_to_preds(preds, active_unit_to_position, observation):
    mask = np.zeros(preds.shape[:-1])
    for unit_id, position in active_unit_to_position.items():
        x, y = position
        resources = _get_unit_resources(unit_id, observation)
        if sum(resources) >= GAME_CONSTANTS['PARAMETERS']['CITY_BUILD_COST']:
            mask[x, y] = 1
    preds[:, :, UNIT_ACTIONS_MAP['bcity']] *= mask
    return preds

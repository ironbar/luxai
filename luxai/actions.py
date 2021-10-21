"""
Functions for generating actions from model predictions
"""
import numpy as np

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


def create_actions_for_units_from_model_predictions(preds, active_units_to_position, units_to_position, action_threshold=0.5):
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
            actions.append(create_unit_action(action_key, unit_id))
            # This ensures that units with overlap do not repeat actions
            preds[x, y, action_idx] = 0
    # TODO: deal with collisions
    return actions


def create_unit_action(action_key, unit_id):
    action_id = action_key.split(' ')[0]
    if action_id == 'm':
        action = 'm %s %s' % (unit_id, action_key.split(' ')[-1])
        return action
    elif action_id in ['bcity', 'p']:
        action = '%s %s' % (action_id, unit_id)
        return action
    elif action_id == 't':
        # TODO: implement transfer
        return 't %s' % unit_id
        # return 'm %s c' % unit_id
    else:
        raise KeyError(action_id)
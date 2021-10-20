"""
Functions for generating actions from model predictions
"""
import numpy as np

from luxai.output_features import CITY_ACTIONS_MAP


def create_actions_for_cities_from_model_predictions(preds, active_cities_to_position, action_threshold=0.5):
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

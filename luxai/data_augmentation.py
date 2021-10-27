"""
Data augmentation
"""
import random
from functools import lru_cache

from luxai.output_features import UNIT_ACTIONS_MAP


def random_data_augmentation(x, y):
    """
    Applies random data augmentation to the given batch
    """
    if random.randint(0, 1):
        x, y = horizontal_flip(x, y)
    return x, y


def horizontal_flip(x, y):
    """
    Horizontal flip on training data
    x is expected to have size (batch_size, 32, 32, 24), (batch_size, 1, 13)
    y is expected to have size (batch_size, 32, 32, unit_actions + 1 ), (batch_size, 32, 32, city_actions + 1)

    I will simply flip the first axis and rearrange the unit action channels. First axis is x, so
    actions involved are east and west
    """
    x = (x[0][:, ::-1], x[1])
    unit_actions_indices = _get_horizontal_flip_unit_actions_indices()
    y = (y[0][:, ::-1, :, unit_actions_indices], y[1][:, ::-1])
    return x, y


@lru_cache(maxsize=1)
def _get_horizontal_flip_unit_actions_indices():
    idx_to_action = {value: key for key, value in UNIT_ACTIONS_MAP.items()}
    def apply_horizontal_flip_to_action(action):
        if action.endswith('e'):
            return action.replace('e', 'w')
        elif action.endswith('w'):
            return action.replace('w', 'e')
        else:
            return action
    indices = [UNIT_ACTIONS_MAP[apply_horizontal_flip_to_action(idx_to_action[idx])] \
        for idx in range(len(UNIT_ACTIONS_MAP))]
    indices.append(len(UNIT_ACTIONS_MAP))
    return indices

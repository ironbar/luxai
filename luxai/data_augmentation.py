"""
Data augmentation
"""
import random
from functools import lru_cache
import numpy as np

from luxai.output_features import UNIT_ACTIONS_MAP


def random_data_augmentation(x, y):
    """
    Applies random data augmentation to the given batch
    """
    if random.randint(0, 1):
        x, y = horizontal_flip(x, y)
    n_rotations = random.randint(0, 3)
    if n_rotations:
        x, y = rotation_90(x, y, n_rotations)
    return x, y


def horizontal_flip(x, y):
    """
    Horizontal flip on training data
    x is expected to have size (batch_size, 32, 32, 24), (batch_size, 1, 13)
    y is expected to have size (batch_size, 32, 32, unit_actions + 1 ), (batch_size, 32, 32, city_actions + 1)

    I will simply flip the first axis and rearrange the unit action channels. First axis is x, so
    actions involved are east and west
    """
    return horizontal_flip_input(x), horizontal_flip_output(y)


def horizontal_flip_input(x):
    x = (x[0][:, ::-1], x[1])
    return x


def horizontal_flip_output(y):
    if len(y) == 4:
        unit_actions_indices = _get_horizontal_flip_unit_actions_indices()[:y[1].shape[-1]]
        y = (y[0][:, ::-1], y[1][:, ::-1, :, unit_actions_indices], y[2][:, ::-1], y[3][:, ::-1])
    elif len(y) == 2:
        unit_actions_indices = _get_horizontal_flip_unit_actions_indices()[:y[0].shape[-1]]
        y = (y[0][:, ::-1, :, unit_actions_indices], y[1][:, ::-1])
    else:
        raise NotImplementedError(len(y))
    return y


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


def rotation_90(x, y, n_times):
    return rotation_90_input(x, n_times), rotation_90_output(y, n_times)


def rotation_90_input(x, n_times):
    x = (np.rot90(x[0], axes=(1, 2), k=n_times), x[1])
    return x


def rotation_90_output(y, n_times):
    if len(y) == 4:
        unit_actions_indices = _get_rotation_unit_actions_indices(n_times)[:y[1].shape[-1]]
        y = (np.rot90(y[0], axes=(1, 2), k=n_times),
             np.rot90(y[1], axes=(1, 2), k=n_times)[:, :, :, unit_actions_indices],
             np.rot90(y[2], axes=(1, 2), k=n_times),
             np.rot90(y[3], axes=(1, 2), k=n_times))
    elif len(y) == 2:
        unit_actions_indices = _get_rotation_unit_actions_indices(n_times)[:y[0].shape[-1]]
        y = (np.rot90(y[0], axes=(1, 2), k=n_times)[:, :, :, unit_actions_indices],
             np.rot90(y[1], axes=(1, 2), k=n_times))
    else:
        raise NotImplementedError(len(y))
    return y


@lru_cache(maxsize=4)
def _get_rotation_unit_actions_indices(n_times):
    indices = (np.arange(4) - n_times) % 4
    indices = indices.tolist() + (indices + 4).tolist() + list(range(8, 11))
    return indices

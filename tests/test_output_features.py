import pytest
import json
import os
import numpy as np

from luxai.input_features import make_input
from luxai.output_features import create_actions_mask, create_output_features

# TODO: once I'm able to generate actions from predictions I can test that by using perfect labels I can
# regenerate the actions for a match
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


@pytest.mark.parametrize('filepath', [
    os.path.join(SCRIPT_DIR, '../data/sample_games/20210924_seed1_172steps.json'),
    os.path.join(SCRIPT_DIR, '../data/sample_games/20210923_seed0_240steps.json'),
])
def test_that_there_are_no_actions_outside_the_masks(filepath):
    with open(filepath, 'r') as f:
        match = json.load(f)

    for step in range(len(match['steps']) - 1):
        observation = match['steps'][step][0]['observation']
        actions = match['steps'][step+1][0]['action'] # notice the step + 150

        ret = make_input(observation)
        active_unit_to_position, active_city_to_position, unit_to_position = ret[2:-1]
        unit_actions_mask = create_actions_mask(active_unit_to_position, observation)
        city_actions_mask = create_actions_mask(active_city_to_position, observation)
        unit_actions, city_actions = create_output_features(actions, unit_to_position, observation)

        _assert_that_there_are_no_actions_outside_the_mask(city_actions, city_actions_mask)
        _assert_that_there_are_no_actions_outside_the_mask(unit_actions, unit_actions_mask)


def _assert_that_there_are_no_actions_outside_the_mask(actions, mask):
    assert (np.max(actions, axis=-1) <= np.max(mask, axis=-1)).all()

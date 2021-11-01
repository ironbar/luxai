import pytest
import os
import json
import numpy as np

from luxai.actions import (
    create_actions_for_cities_from_model_predictions, CITY_ACTIONS_MAP,
    create_actions_for_units_from_model_predictions, UNIT_ACTIONS_MAP,
    get_blocked_positions_using_units_that_do_not_move,
    rank_units_based_on_priority, remove_collision_actions)
from luxai.input_features import make_input
from luxai.output_features import create_output_features

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


@pytest.mark.parametrize('player', range(2))
@pytest.mark.parametrize('filepath', [
    os.path.join(SCRIPT_DIR, '../data/sample_games/20210924_seed1_172steps.json'),
    os.path.join(SCRIPT_DIR, '../data/sample_games/20210923_seed0_240steps.json'),
])
def test_city_actions_can_be_recovered_from_ground_truth(filepath, player):
    with open(filepath, 'r') as f:
        match = json.load(f)['steps']
    for observation, actions in step_generator(match, player):
        ret = make_input(observation)
        active_city_to_position, unit_to_position, city_to_position = ret[3:]
        _, city_actions = create_output_features(actions, unit_to_position, observation)
        true_city_actions = [action for action in actions if action.split(' ')[0] in CITY_ACTIONS_MAP]

        recovered_actions = create_actions_for_cities_from_model_predictions(
            city_actions, active_city_to_position, len(city_to_position) - len(unit_to_position),
            is_post_processing_enabled=False)

        msg = 'actions: %s\ncity actions: %s\nrecovered city actions: %s' % (actions, true_city_actions, recovered_actions)
        if recovered_actions:
            assert all(action in actions for action in recovered_actions), msg
        if true_city_actions:
            assert all(action in recovered_actions for action in true_city_actions), msg


@pytest.mark.parametrize('player', range(2))
@pytest.mark.parametrize('filepath', [
    os.path.join(SCRIPT_DIR, '../data/sample_games/20210924_seed1_172steps.json'),
    os.path.join(SCRIPT_DIR, '../data/sample_games/20210923_seed0_240steps.json'),
])
def test_unit_actions_can_be_recovered_from_ground_truth_if_no_unit_overlaps_and_skipping_transfer(filepath, player):
    with open(filepath, 'r') as f:
        match = json.load(f)['steps']
    for observation, actions in step_generator(match, player):
        active_unit_to_position, _, unit_to_position, city_to_position = make_input(observation)[2:]
        unit_actions_ground_truth, _ = create_output_features(actions, unit_to_position, observation)

        recovered_actions = create_actions_for_units_from_model_predictions(
            unit_actions_ground_truth, active_unit_to_position, unit_to_position, observation,
            set(city_to_position.keys()), is_post_processing_enabled=False)

        units_with_overlap = _get_units_with_overlap(unit_to_position)
        true_unit_actions = _remove_actions_with_overlap_or_transfer(actions, units_with_overlap)
        recovered_actions = _remove_actions_with_overlap_or_transfer(recovered_actions, units_with_overlap)

        msg = 'actions:  %s\ntrue unit actions:      %s\nrecovered unit actions: %s' % (
            sorted(actions), sorted(true_unit_actions), sorted(recovered_actions))
        msg += '\nunits with overlap: %s' % units_with_overlap
        if recovered_actions:
            assert all(action in actions for action in recovered_actions), msg
        if true_unit_actions:
            assert all(action in recovered_actions for action in true_unit_actions), msg


def _remove_actions_with_overlap_or_transfer(actions, units_with_overlap):
    true_unit_actions = [action for action in actions if action.split(' ')[0] in set(['m', 'bcity', 'p'])]
    true_unit_actions = [action for action in true_unit_actions if not action.endswith(' c')]
    true_unit_actions = [action for action in true_unit_actions if not any(_is_action_from_unit(action, unit_id) for unit_id in units_with_overlap)]
    return true_unit_actions


def _get_units_with_overlap(unit_to_position):
    units_with_overlap = []
    for unit_id, position in unit_to_position.items():
        for other_unit_id, other_position in unit_to_position.items():
            if unit_id == other_unit_id:
                continue
            if position == other_position:
                units_with_overlap.append(unit_id)
    return set(units_with_overlap)


def _is_action_from_unit(action, unit_id):
    return ' %s ' % unit_id in action or action.endswith(unit_id)


def step_generator(match, player):
    for step in range(len(match) - 1):
        observation = match[step][0]['observation']
        if player:
            observation.update(match[step][player]['observation'])
        actions = match[step+1][player]['action'] # notice the step + 1
        if actions is None: # this can happen on timeout
            continue
        yield observation, actions


@pytest.mark.parametrize('player', range(2))
@pytest.mark.parametrize('filepath', [
    os.path.join(SCRIPT_DIR, '../data/sample_games/20210924_seed1_172steps.json'),
])
def test_recovered_actions_from_units_with_overlap_has_same_length_as_ground_truth(filepath, player):
    with open(filepath, 'r') as f:
        match = json.load(f)['steps']
    for observation, actions in step_generator(match, player):
        active_unit_to_position, _, unit_to_position, city_to_position = make_input(observation)[2:]
        unit_actions_ground_truth, _ = create_output_features(actions, unit_to_position, observation)

        recovered_actions = create_actions_for_units_from_model_predictions(
            unit_actions_ground_truth, active_unit_to_position, unit_to_position, observation,
            set(city_to_position.keys()), is_post_processing_enabled=False)

        units_with_overlap = _get_units_with_overlap(unit_to_position)
        true_unit_actions = _remove_actions_without_overlap_or_with_transfer(actions, units_with_overlap)
        recovered_actions = _remove_actions_without_overlap_or_with_transfer(recovered_actions, units_with_overlap)

        msg = 'actions:  %s\ntrue unit actions:      %s\nrecovered unit actions: %s' % (
            sorted(actions), sorted(true_unit_actions), sorted(recovered_actions))
        msg += '\nunits with overlap: %s' % units_with_overlap
        assert len(true_unit_actions) == len(recovered_actions), msg


def _remove_actions_without_overlap_or_with_transfer(actions, units_with_overlap):
    true_unit_actions = [action for action in actions if action.split(' ')[0] in set(['m', 'bcity', 'p'])]
    true_unit_actions = [action for action in true_unit_actions if not action.endswith(' c')]
    true_unit_actions = [action for action in true_unit_actions if any(_is_action_from_unit(action, unit_id) for unit_id in units_with_overlap)]
    return true_unit_actions


@pytest.mark.parametrize('player', [0])
@pytest.mark.parametrize('filepath', [
    os.path.join(SCRIPT_DIR, '../data/sample_games/20210924_seed1_172steps.json'),
    os.path.join(SCRIPT_DIR, '../data/sample_games/20210923_seed0_240steps.json'),
])
def test_transfer_actions_can_be_recovered_from_ground_truth_if_no_unit_overlaps(filepath, player):
    with open(filepath, 'r') as f:
        match = json.load(f)['steps']
    for observation, actions in step_generator(match, player):
        active_unit_to_position, _, unit_to_position, city_to_position = make_input(observation)[2:]
        unit_actions_ground_truth, _ = create_output_features(actions, unit_to_position, observation)

        recovered_actions = create_actions_for_units_from_model_predictions(
            unit_actions_ground_truth, active_unit_to_position, unit_to_position, observation, set(city_to_position.keys()))

        units_with_overlap = _get_units_with_overlap(unit_to_position)
        true_unit_actions = _remove_actions_with_overlap_or_without_transfer(actions, units_with_overlap)
        recovered_actions = _remove_actions_with_overlap_or_without_transfer(recovered_actions, units_with_overlap)

        msg = 'actions:  %s\ntrue unit actions:      %s\nrecovered unit actions: %s' % (
            sorted(actions), sorted(true_unit_actions), sorted(recovered_actions))
        msg += '\nunits with overlap: %s' % units_with_overlap
        if recovered_actions:
            assert all(action in actions for action in recovered_actions), msg
        if true_unit_actions:
            assert all(action in recovered_actions for action in true_unit_actions), msg


def _remove_actions_with_overlap_or_without_transfer(actions, units_with_overlap):
    true_unit_actions = [action for action in actions if action.split(' ')[0]== 't']
    true_unit_actions = [action for action in true_unit_actions if not any(_is_action_from_unit(action, unit_id) for unit_id in units_with_overlap)]
    return true_unit_actions

@pytest.mark.parametrize('unit_to_position, unit_to_action, city_positions, blocked_positions', [
    ({0: (0, 0)}, {0: 'bcity'}, set(), set([(0, 0)])), # unit that is going to build a city is added to blocked positions
    ({0: (0, 0)}, {0: 't '}, set([(0, 0)]), set()), # unit on a building that does not move is not added to blocked positions
    ({0: (0, 0)}, {0: 'm n'}, set(), set()), # unit that moves is not added to blocked positions
    ({0: (0, 0)}, {}, set(), set([(0, 0)])), # unit that that not has action is added to blocked positions
])
def test_get_blocked_positions_using_units_that_do_not_move(unit_to_position, unit_to_action, city_positions, blocked_positions):
    assert blocked_positions == get_blocked_positions_using_units_that_do_not_move(unit_to_position, unit_to_action, city_positions)

@pytest.mark.parametrize('unit_to_priority, ranked_units', [
    ({0: 0, 1: 1}, [1, 0]),
    ({0: 1, 1: 0}, [0, 1]),
    ({}, []),
])
def test_rank_units_based_on_priority(unit_to_priority, ranked_units):
    assert ranked_units == rank_units_based_on_priority(unit_to_priority)


@pytest.mark.parametrize('unit_to_action, unit_to_position, unit_to_priority, city_positions, remaining_actions', [
    ({0: 'm 0 n'}, {0: (1, 1)}, {0: 1}, set(), {0: 'm 0 n'}), # no obstacle
    ({0: 'm 0 n'}, {0: (1, 1), 1: (1, 0)}, {0: 1}, set(), {}), # a unit does not allow to move
    ({0: 'm 0 n'}, {0: (1, 1), 1: (1, 0)}, {0: 1}, set([(1, 0)]), {0: 'm 0 n'}), # the unit is on a city so move is allowed
    ({0: 'm 0 s'}, {0: (1, 1), 1: (1, 0)}, {0: 1}, set(), {0: 'm 0 s'}), # moving in the other direction is allowed
])
def test_remove_collision_actions(unit_to_action, unit_to_position, unit_to_priority, city_positions, remaining_actions):
    remove_collision_actions(unit_to_action, unit_to_position, unit_to_priority, city_positions)
    assert remaining_actions == unit_to_action

@pytest.mark.parametrize('build_action', ['bw', 'bc'])
@pytest.mark.parametrize('empty_unit_slots', list(range(5)))
def test_build_unit_actions_are_removed_if_no_empty_slots_are_avaible(empty_unit_slots, build_action):
    preds = np.zeros((2, 2, len(CITY_ACTIONS_MAP)))
    preds[:, :, CITY_ACTIONS_MAP[build_action]] = 1
    active_city_to_position = {}
    for x in range(2):
        for y in range(2):
            active_city_to_position['%i_%i' % (x, y)] = (x, y)
    actions = create_actions_for_cities_from_model_predictions(
        preds, active_city_to_position, empty_unit_slots)
    assert empty_unit_slots == len(actions)


@pytest.mark.parametrize('empty_unit_slots', list(range(5)))
def test_bw_actions_are_replaced_by_research_if_no_empty_slots_are_avaible(empty_unit_slots):
    preds = np.ones((2, 2, len(CITY_ACTIONS_MAP)))
    active_city_to_position = {}
    for x in range(2):
        for y in range(2):
            active_city_to_position['%i_%i' % (x, y)] = (x, y)
    actions = create_actions_for_cities_from_model_predictions(
        preds, active_city_to_position, empty_unit_slots)
    assert len(active_city_to_position) == len(actions)

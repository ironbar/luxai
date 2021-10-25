"""
Functions for dealing with data
"""
import os
import json
import numpy as np
from tqdm import tqdm
import pandas as pd

from luxai.input_features import make_input, expand_board_size_adding_zeros
from luxai.output_features import create_actions_mask, create_output_features


def load_train_and_test_data(n_matches, test_fraction, matches_json_dir,
                             matches_cache_npz_dir, agent_selection_path):
    matches = load_best_n_matches(
        n_matches=n_matches, matches_json_dir=matches_json_dir,
        matches_cache_npz_dir=matches_cache_npz_dir, agent_selection_path=agent_selection_path)

    test_matches = [match for idx, match in enumerate(matches) if not idx%test_fraction]
    train_matches = [match for idx, match in enumerate(matches) if idx%test_fraction]

    print('Train matches: %i' % len(train_matches))
    train_data = combine_data_for_training(train_matches)
    print('Test matches: %i' % len(test_matches))
    test_data = combine_data_for_training(test_matches)
    # test data is shuffle to avoid nan in metrics because at the end there might be long periods without actions
    test_data = shuffle_data(test_data)

    return train_data, test_data


def load_best_n_matches(n_matches, matches_json_dir, matches_cache_npz_dir, agent_selection_path):
    df = pd.read_csv(agent_selection_path)
    df.sort_values('FinalScore', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    matches = []
    for episode_id, player in tqdm(zip(df.EpisodeId[:n_matches], df.Index[:n_matches]), total=n_matches, desc='Loading matches'):
        npz_filepath = os.path.join(matches_cache_npz_dir, '%i_%i.npz' % (episode_id, player))

        if os.path.exists(npz_filepath):
            match = load_match_from_npz(npz_filepath)
        else:
            json_filepath = os.path.join(matches_json_dir, '%i.json' % episode_id)
            match = load_match_from_json(json_filepath, player)
            save_match_to_npz(npz_filepath, match)

        matches.append(match)
    return matches


def combine_data_for_training(matches):
    inputs = [
        np.concatenate([expand_board_size_adding_zeros(match['board']) for match in matches]),
        np.concatenate([match['features'] for match in matches]),
    ]
    print('Inputs shapes', [x.shape for x in inputs])
    outputs = [
        np.concatenate([expand_board_size_adding_zeros(match['unit_output']) for match in matches]),
        np.concatenate([expand_board_size_adding_zeros(match['city_output']) for match in matches]),
    ]
    print('Outputs shapes', [x.shape for x in outputs])
    return inputs, outputs


def shuffle_data(data):
    new_order = np.arange(len(data[0][0]))
    np.random.shuffle(new_order)
    return [x[new_order] for x in data[0]], [x[new_order] for x in data[1]]



def load_match_from_json(filepath, player):
    """
    Given the path to a match saved on json and the player it loads data for training

    Returns
    --------
    dict(board=board, features=features, unit_output=unit_output, city_output=city_output)
    """
    with open(filepath, 'r') as f:
        match = json.load(f)

    board, features, unit_output, city_output = [], [], [], []
    for step in range(len(match) - 1):
        observation = match[step][0]['observation']
        if player:
            observation.update(match[step][player]['observation'])
        actions = match[step+1][player]['action'] # notice the step + 150
        if actions is None: # this can happen on timeout
            continue

        ret = make_input(observation)
        active_units_to_position, active_cities_to_position, units_to_position = ret[2:-1]
        if active_units_to_position or active_cities_to_position:
            board.append(ret[0])
            features.append(ret[1])
            unit_actions_mask = create_actions_mask(active_units_to_position, observation)
            city_actions_mask = create_actions_mask(active_cities_to_position, observation)
            unit_actions, city_actions = create_output_features(actions, units_to_position, observation)
            unit_output.append(np.concatenate([unit_actions, unit_actions_mask], axis=-1))
            city_output.append(np.concatenate([city_actions, city_actions_mask], axis=-1))

    board = np.array(board, dtype=np.float32)
    features = np.array(features, dtype=np.float32)
    unit_output = np.array(unit_output, dtype=np.float32)
    city_output = np.array(city_output, dtype=np.float32)
    return dict(board=board, features=features, unit_output=unit_output, city_output=city_output)


def save_match_to_npz(filepath, match):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez_compressed(filepath, **match)


def load_match_from_npz(filepath):
    return dict(**np.load(filepath))



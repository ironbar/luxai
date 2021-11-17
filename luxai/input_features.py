"""
Input features for training imitation learning model

The start point is https://www.kaggle.com/shoheiazuma/lux-ai-with-imitation-learning
"""
import numpy as np

from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game_constants import GAME_CONSTANTS


IDENTIFIER_TO_OBJECT = {
    'u': 'unit',
    'ct': 'city_tile',
    'r': 'resource',
    'rp': 'research',
    'c': 'city',
    'ccd': 'road',
}


CHANNELS_MAP = dict(
    wood=0, coal=1, uranium=2,
    player_worker=3, player_cart=4, player_city=5,
    opponent_worker=6, opponent_cart=7, opponent_city=8,
    cooldown=9, road_level=10,
    player_city_fuel=11, opponent_city_fuel=12,
    player_unit_cargo=13, opponent_unit_cargo=14,
    player_unit_fuel=15, opponent_unit_fuel=16,
    player_city_can_survive_next_night=17, opponent_city_can_survive_next_night=18,
    player_city_can_survive_until_end=19, opponent_city_can_survive_until_end=20,
    resources_available=21, fuel_available=22,
    player_is_unit_full=23, is_cell_emtpy=24, player_can_build_city=25,
    player_obstacles=26, playable_area=27,
)


FEATURES_MAP = dict(
    step=0, is_night=1, is_last_day=2,
    player_research_points=3, opponent_research_points=4,
    is_player_in_coal_era=5, is_player_in_uranium_era=6,
    is_opponent_in_coal_era=7, is_opponent_in_uranium_era=8,
    # player_n_cities=9, player_n_units=10,
    # opponent_n_cities=11, opponent_n_units=12,
    # hour=13, city_diff=14,
    # unit_free_slots=15,
    hour=9, city_diff=10,
    unit_free_slots=11,
)


def make_input(obs):
    """
    Creates 3d board and 1d features that can be used as input to a model
    Values are normalized to avoid having quantities much bigger than one

    It also computes some dictionaries that could be later used to create the output for the model

    Returns
    -------
    board, features, active_unit_to_position, active_city_to_position, unit_to_position
    """
    width, height = obs['width'], obs['height']
    city_id_to_survive_nights = {}

    board = np.zeros((len(CHANNELS_MAP), width, height), dtype=np.float32)
    features = np.zeros(len(FEATURES_MAP), dtype=np.float32)
    active_unit_to_position, active_city_to_position = dict(), dict()
    unit_to_position, city_to_position = dict(), dict()

    for update in obs['updates']:
        splits = update.split(' ')
        input_identifier = splits[0]
        object_type = IDENTIFIER_TO_OBJECT.get(input_identifier, None)

        if object_type == 'unit':
            unit_type, team, unit_id, x, y, cooldown, wood, coal, uranium = parse_unit_info(splits)
            prefix = get_prefix_for_channels_map(team, obs)
            if is_worker(unit_type):
                board[CHANNELS_MAP['%s_worker' % prefix], x, y] += 1
            else:
                board[CHANNELS_MAP['%s_cart' % prefix], x, y] += 1
            board[CHANNELS_MAP['%s_unit_cargo' % prefix], x, y] = get_normalized_cargo(unit_type, wood, coal, uranium)
            board[CHANNELS_MAP['%s_unit_fuel' % prefix], x, y] = get_normalized_unit_fuel(unit_type, wood, coal, uranium)
            board[CHANNELS_MAP['cooldown'], x, y] = normalize_cooldown(cooldown)
            if prefix == 'player':
                unit_to_position[unit_id] = (x, y)
                if cooldown < 1:
                    active_unit_to_position[unit_id] = (x, y)
        elif object_type == 'city_tile':
            team, city_id, x, y, cooldown = parse_city_tile_info(splits)
            prefix = get_prefix_for_channels_map(team, obs)
            board[CHANNELS_MAP['%s_city' % prefix], x, y] = 1
            board[CHANNELS_MAP['%s_city_fuel' % prefix], x, y] = city_id_to_survive_nights[city_id]
            board[CHANNELS_MAP['%s_city_can_survive_next_night' % prefix], x, y] = \
                city_id_to_survive_nights[city_id] > (10 - max(obs['step'] % 40 - 30, 0))/10
            board[CHANNELS_MAP['%s_city_can_survive_until_end' % prefix], x, y] = \
                city_id_to_survive_nights[city_id] > (360 - obs['step'] ) // 40 + (10 - max(obs['step'] % 40 - 30, 0))/10
            board[CHANNELS_MAP['cooldown'], x, y] = normalize_cooldown(cooldown)
            if prefix == 'player':
                city_to_position['%s_%i' % (city_id, len(city_to_position))] = (x, y)
                if cooldown < 1:
                    active_city_to_position['%s_%i' % (city_id, len(active_city_to_position))] = (x, y)
        elif object_type == 'resource':
            resource_type, x, y, amount = parse_resource_info(splits)
            board[CHANNELS_MAP[resource_type], x, y] = amount / 800
        elif object_type == 'research':
            team, research_points = parse_research_points_info(splits)
            prefix = get_prefix_for_channels_map(team, obs)
            features[FEATURES_MAP['%s_research_points' % prefix]] = \
                research_points / GAME_CONSTANTS['PARAMETERS']['RESEARCH_REQUIREMENTS']['URANIUM']
            features[FEATURES_MAP['is_%s_in_coal_era' % prefix]] = \
                research_points >= GAME_CONSTANTS['PARAMETERS']['RESEARCH_REQUIREMENTS']['COAL']
            features[FEATURES_MAP['is_%s_in_uranium_era' % prefix]] = \
                research_points >= GAME_CONSTANTS['PARAMETERS']['RESEARCH_REQUIREMENTS']['URANIUM']
        elif object_type == 'city':
            team, city_id, fuel, lightupkeep = parse_city_info(splits)
            city_id_to_survive_nights[city_id] = fuel / lightupkeep / 10 # number of nights a city can survive (a night is 10 steps)
        elif object_type == 'road':
            x, y, road_level = parse_road_info(splits)
            board[CHANNELS_MAP['road_level'], x, y] = road_level/6

    board[CHANNELS_MAP['playable_area']] = 1
    board[CHANNELS_MAP['player_is_unit_full']] = board[CHANNELS_MAP['player_unit_cargo']] == 1
    board[CHANNELS_MAP['is_cell_emtpy']] = _is_cell_empty(board)
    board[CHANNELS_MAP['player_can_build_city']] = _player_can_build_city(board)
    board[CHANNELS_MAP['player_obstacles']] = _player_obstacles(board)
    add_resources_and_fuel_available_to_gather(board, features)

    features[FEATURES_MAP['step']] = obs['step'] / 360
    features[FEATURES_MAP['is_night']] = obs['step'] % 40 >= 30
    features[FEATURES_MAP['is_last_day']] = obs['step'] >= 40*8
    features[FEATURES_MAP['hour']] = obs['step'] % 40 / 40
    features[FEATURES_MAP['city_diff']] = np.sum(board[CHANNELS_MAP['player_city']]) - np.sum(board[CHANNELS_MAP['opponent_city']])
    features[FEATURES_MAP['unit_free_slots']] = np.sum(board[CHANNELS_MAP['player_city']]) - np.sum(board[CHANNELS_MAP['player_worker']]) - np.sum(board[CHANNELS_MAP['player_cart']])
    # for prefix in ['player', 'opponent']:
    #     # Features are divided by 10 to avoid very big numbers
    #     features[FEATURES_MAP['%s_n_cities' % prefix]] = np.sum(board[CHANNELS_MAP['%s_city' % prefix]])/10
    #     features[FEATURES_MAP['%s_n_units' % prefix]] += np.sum(board[CHANNELS_MAP['%s_worker' % prefix]])/10
    #     features[FEATURES_MAP['%s_n_units' % prefix]] += np.sum(board[CHANNELS_MAP['%s_cart' % prefix]])/10

    board = np.transpose(board, axes=(1, 2, 0))
    features = np.expand_dims(features, axis=0)

    return board, features, active_unit_to_position, active_city_to_position, unit_to_position, city_to_position


def parse_unit_info(splits):
    unit_type = int(splits[1])
    team = int(splits[2])
    unit_id = splits[3]
    x = int(splits[4])
    y = int(splits[5])
    cooldown = float(splits[6])
    wood = int(splits[7])
    coal = int(splits[8])
    uranium = int(splits[9])
    return unit_type, team, unit_id, x, y, cooldown, wood, coal, uranium


def parse_city_tile_info(splits):
    team = int(splits[1])
    city_id = splits[2]
    x = int(splits[3])
    y = int(splits[4])
    cooldown = float(splits[5])
    return team, city_id, x, y, cooldown


def parse_resource_info(splits):
    resource_type = splits[1]
    x = int(splits[2])
    y = int(splits[3])
    amount = int(float(splits[4]))
    return resource_type, x, y, amount


def parse_research_points_info(splits):
    team = int(splits[1])
    research_points = int(splits[2])
    return team, research_points


def parse_city_info(splits):
    team = int(splits[1])
    city_id = splits[2]
    fuel = float(splits[3])
    lightupkeep = float(splits[4])
    return team, city_id, fuel, lightupkeep


def parse_road_info(splits):
    x = int(splits[1])
    y = int(splits[2])
    road_level = float(splits[3])
    return x, y, road_level


def get_prefix_for_channels_map(team, obs):
    if team == obs['player']:
        prefix = 'player'
    else:
        prefix = 'opponent'
    return prefix


def is_worker(unit_type):
    return unit_type == 0


def get_normalized_cargo(unit_type, wood, coal, uranium):
    """
    Returns a value between 0 and 1 where 0 means the unit has no cargo, and 1 means that the unit
    is full
    """
    cargo = wood + coal + uranium
    if is_worker(unit_type):
        cargo /= GAME_CONSTANTS['PARAMETERS']['RESOURCE_CAPACITY']['WORKER']
    else:
        cargo /= GAME_CONSTANTS['PARAMETERS']['RESOURCE_CAPACITY']['CART']
    return cargo


def get_normalized_unit_fuel(unit_type, wood, coal, uranium):
    """
    Returns a value between 0 and 1 where 0 means the unit has no fuel, and 1 means that the unit
    is full with uranium
    """
    fuel_rate = GAME_CONSTANTS['PARAMETERS']['RESOURCE_TO_FUEL_RATE']
    resource_capacity = GAME_CONSTANTS['PARAMETERS']['RESOURCE_CAPACITY']

    fuel = wood*fuel_rate['WOOD'] \
        + coal*fuel_rate['COAL'] \
        + uranium*fuel_rate['URANIUM']
    if is_worker(unit_type):
        fuel /= resource_capacity['WORKER']
    else:
        fuel /= resource_capacity['CART']
    fuel /= fuel_rate['URANIUM']
    return fuel


def normalize_cooldown(cooldown):
    return (cooldown - 1)/10


def add_resources_and_fuel_available_to_gather(board, features):
    collection_rate = GAME_CONSTANTS['PARAMETERS']['WORKER_COLLECTION_RATE']
    fuel_rate = GAME_CONSTANTS['PARAMETERS']['RESOURCE_TO_FUEL_RATE']

    board[CHANNELS_MAP['resources_available']] += (board[CHANNELS_MAP['wood']] > 0)*collection_rate['WOOD']
    if features[FEATURES_MAP['is_player_in_coal_era']]:
        board[CHANNELS_MAP['resources_available']] += (board[CHANNELS_MAP['coal']] > 0)*collection_rate['COAL']
    if features[FEATURES_MAP['is_player_in_uranium_era']]:
        board[CHANNELS_MAP['resources_available']] += (board[CHANNELS_MAP['uranium']] > 0)*collection_rate['URANIUM']
    _expand_available_resource(board[CHANNELS_MAP['resources_available']])
    board[CHANNELS_MAP['resources_available']] /= collection_rate['WOOD']*5

    board[CHANNELS_MAP['fuel_available']] += (board[CHANNELS_MAP['wood']] > 0)*collection_rate['WOOD']*fuel_rate['WOOD']
    if features[FEATURES_MAP['is_player_in_coal_era']]:
        board[CHANNELS_MAP['fuel_available']] += (board[CHANNELS_MAP['coal']] > 0)*collection_rate['COAL']*fuel_rate['COAL']
    if features[FEATURES_MAP['is_player_in_uranium_era']]:
        board[CHANNELS_MAP['fuel_available']] += (board[CHANNELS_MAP['uranium']] > 0)*collection_rate['URANIUM']*fuel_rate['URANIUM']
    _expand_available_resource(board[CHANNELS_MAP['fuel_available']])
    board[CHANNELS_MAP['fuel_available']] /= collection_rate['URANIUM']*fuel_rate['URANIUM']*5


def _expand_available_resource(channel):
    channel_original = channel.copy()
    channel[:-1] += channel_original[1:]
    channel[1:] += channel_original[:-1]
    channel[:, :-1] += channel_original[:, 1:]
    channel[:, 1:] += channel_original[:, :-1]


def _is_cell_empty(board):
    is_cell_used = board[CHANNELS_MAP['player_city']] + board[CHANNELS_MAP['opponent_city']]
    is_cell_used += board[CHANNELS_MAP['wood']] + board[CHANNELS_MAP['coal']] + board[CHANNELS_MAP['uranium']]
    return is_cell_used == 0


def _player_can_build_city(board):
    can_build_city = board[CHANNELS_MAP['player_is_unit_full']].copy()
    can_build_city *= board[CHANNELS_MAP['is_cell_emtpy']]
    can_build_city *= np.clip(board[CHANNELS_MAP['player_worker']], None, 1)
    can_build_city *= (board[CHANNELS_MAP['cooldown']] < 0)
    return can_build_city


def _player_obstacles(board):
    obstacles = board[CHANNELS_MAP['opponent_city']]
    frozen_units = board[CHANNELS_MAP['cooldown']] >= 0
    for key in ['player', 'opponent']:
        units = board[CHANNELS_MAP['%s_worker' % key]] + board[CHANNELS_MAP['%s_cart' % key]]
        obstacles += (1 - board[CHANNELS_MAP['%s_city' % key]])*units*frozen_units
    return obstacles


def expand_board_size_adding_zeros(board, size=32):
    """
    Increases the board size by default to 32x32 by adding zeros. The input should be a 4d array
    """
    board_size = board.shape[1]
    if board_size < size:
        expanded_board = np.zeros((board.shape[0], size, size, board.shape[3]), dtype=board.dtype)
        offset = (size - board_size)//2
        expanded_board[:, offset:-offset, offset:-offset] = board
        return expanded_board
    elif board_size == size:
        return board
    else:
        raise NotImplementedError()


def crop_board_to_original_size(board, observation):
    """
    Increases the board size by default to 32x32 by adding zeros. The input should be a 4d array
    """
    original_size = observation['width']
    board_size = board.shape[1]
    if board_size > original_size:
        offset = (board_size - original_size)//2
        return board[:, offset:-offset, offset:-offset]
    elif board_size == original_size:
        return board
    else:
        raise NotImplementedError()
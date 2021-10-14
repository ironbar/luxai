"""
Input and output features for training imitation learning model

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
)


FEATURES_MAP = dict(
    step=0, is_night=1, is_last_day=2,
    player_research_points=3, opponent_research_points=4,
    is_player_in_coal_era=5, is_player_in_uranium_era=6,
    is_opponent_in_coal_era=7, is_opponent_in_uranium_era=8,
    player_n_cities=9, player_n_units=10,
    opponent_n_cities=11, opponent_n_units=12,
)


def make_input(obs):
    width, height = obs['width'], obs['height']
    city_id_to_survive_turns = {}

    board = np.zeros((len(CHANNELS_MAP), height, width), dtype=np.float32)
    features = np.zeros(len(FEATURES_MAP), dtype=np.float32)

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
            board[CHANNELS_MAP['cooldown'], x, y] = cooldown
        elif object_type == 'city_tile':
            team, city_id, x, y, cooldown = parse_city_tile_info(splits)
            prefix = get_prefix_for_channels_map(team, obs)
            board[CHANNELS_MAP['%s_city' % prefix], x, y] = 1
            board[CHANNELS_MAP['%s_city_fuel' % prefix], x, y] = city_id_to_survive_turns[city_id]
            board[CHANNELS_MAP['cooldown'], x, y] = cooldown
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
            city_id_to_survive_turns[city_id] = fuel / lightupkeep
        elif object_type == 'road':
            x, y, road_level = parse_road_info(splits)
            board[CHANNELS_MAP['road_level'], x, y] = road_level

    # # Day/Night Cycle
    # board[17, :] = obs['step'] % 40 / 40
    # # Turns
    # board[18, :] = obs['step'] / 360
    # # Map Size
    # board[19, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1

    return board, features


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
    fuel = wood*GAME_CONSTANTS['PARAMETERS']['RESOURCE_TO_FUEL_RATE']['WOOD'] \
        + coal*GAME_CONSTANTS['PARAMETERS']['RESOURCE_TO_FUEL_RATE']['COAL'] \
        + uranium*GAME_CONSTANTS['PARAMETERS']['RESOURCE_TO_FUEL_RATE']['URANIUM']
    if is_worker(unit_type):
        fuel /= GAME_CONSTANTS['PARAMETERS']['RESOURCE_CAPACITY']['WORKER']
    else:
        fuel /= GAME_CONSTANTS['PARAMETERS']['RESOURCE_CAPACITY']['CART']
    fuel /= GAME_CONSTANTS['PARAMETERS']['RESOURCE_TO_FUEL_RATE']['URANIUM']
    return fuel

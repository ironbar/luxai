import math
from typing import List

from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game_constants import GAME_CONSTANTS
from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.constants import Constants
from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game import Game
from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game_objects import Player, Unit, CityTile
from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game_map import Cell, Position

def get_resource_tiles(game_state: Game) -> List[Cell]:
    """ Returns a list with all the Cells that have resources in the map """
    resource_tiles: List[Cell] = []
    for y in range(game_state.map.height):
        for x in range(game_state.map.width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource():
                resource_tiles.append(cell)
    return resource_tiles


def get_empty_tiles(game_state: Game) -> List[Cell]:
    """ Returns a list with all the Cells that do not have resources nor cities """
    empty_tiles: List[Cell] = []
    for y in range(game_state.map.height):
        for x in range(game_state.map.width):
            cell = game_state.map.get_cell(x, y)
            if not cell.has_resource() and cell.citytile is None:
                empty_tiles.append(cell)
    return empty_tiles


def get_available_units(player: Player) -> List[Unit]:
    """ Returns a list with the workers that are available to do actions """
    return [unit for unit in player.units if unit.can_act()]


def get_available_workers(player: Player) -> List[Unit]:
    """ Returns a list with the workers that are available to do actions """
    return [unit for unit in player.units if unit.is_worker() and unit.can_act()]


def get_non_available_workers(player: Player) -> List[Unit]:
    """ Returns a list with the workers that are not available and will stay on same tile """
    return [unit for unit in player.units if unit.is_worker() and not unit.can_act()]


def get_available_city_tiles(player: Player) -> List[CityTile]:
    return [city_tile for city_tile in get_all_city_tiles(player) if city_tile.can_act()]


def get_all_city_tiles(player: Player) -> List[CityTile]:
    city_tiles = []
    for _, city in player.cities.items():
        city_tiles.extend(city.citytiles)
    return city_tiles


def get_n_buildable_units(player: Player) -> int:
    return len(get_all_city_tiles(player)) - len(player.units)


def move_to_closest_resource(unit: Unit, player: Player, resource_tiles: List[Cell]) -> str:
    """ Moves the unit towards the closest resource, returns None if there is no available resource """
    closest_resource_tile = find_closest_resource(unit, player, resource_tiles)
    if closest_resource_tile is not None:
        return unit.move(unit.pos.direction_to(closest_resource_tile.pos))

def find_closest_resource(unit: Unit, player: Player, resource_tiles: List[Cell]) -> Cell:
    closest_dist = math.inf
    closest_resource_tile = None
    for resource_tile in resource_tiles:
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.COAL and not player.researched_coal(): continue
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.URANIUM and not player.researched_uranium(): continue
        dist = resource_tile.pos.distance_to(unit.pos)
        if dist < closest_dist:
            closest_dist = dist
            closest_resource_tile = resource_tile
    return closest_resource_tile


def move_to_closest_city_tile(unit: Unit, player: Player) -> str:
    """ Moves the unit towards the closest city tile, returns None if there is no available resource """
    closest_city_tile = find_closest_city_tile(unit, player)
    if closest_city_tile is not None:
        move_dir = unit.pos.direction_to(closest_city_tile.pos)
        return unit.move(move_dir)

def find_closest_city_tile(unit: Unit, player: Player) -> CityTile:
    closest_dist = math.inf
    closest_city_tile = None
    for _, city in player.cities.items():
        for city_tile in city.citytiles:
            dist = city_tile.pos.distance_to(unit.pos)
            if dist < closest_dist:
                closest_dist = dist
                closest_city_tile = city_tile
    return closest_city_tile


def find_closest_tile_to_unit(unit: Unit, candidate_tiles: List[Cell]) -> Cell:
    closest_dist = math.inf
    closest_tile = None
    for tile in candidate_tiles:
        dist = tile.pos.distance_to(unit.pos)
        if dist < closest_dist:
            closest_dist = dist
            closest_tile = tile
    return closest_tile


def is_position_in_list(position, positions):
    return any(position.equals(other_position) for other_position in positions)


def get_directions_to(source: Position, target: Position):
    directions = []
    x_diff = target.x - source.x
    if x_diff < 0:
        directions.append(Constants.DIRECTIONS.WEST)
    elif x_diff > 0:
        directions.append(Constants.DIRECTIONS.EAST)

    y_diff = target.y - source.y
    if y_diff < 0:
        directions.append(Constants.DIRECTIONS.NORTH)
    elif y_diff > 0:
        directions.append(Constants.DIRECTIONS.SOUTH)

    if directions:
        return directions
    else:
        return [Constants.DIRECTIONS.CENTER]


def get_unit_cargo(unit):
    cargo = GAME_CONSTANTS['PARAMETERS']['RESOURCE_CAPACITY']['CART']*unit.type \
        + GAME_CONSTANTS['PARAMETERS']['RESOURCE_CAPACITY']['WORKER']*(1 - unit.type) \
        - unit.get_cargo_space_left()
    return cargo

def is_cart(unit):
    return unit.type

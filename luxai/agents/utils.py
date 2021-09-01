import math

from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.constants import Constants


def get_resource_tiles(game_state):
    resource_tiles: list[Cell] = []
    for y in range(game_state.map.height):
        for x in range(game_state.map.width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource():
                resource_tiles.append(cell)
    return resource_tiles


def get_available_workers(player):
    return [unit for unit in player.units if unit.is_worker() and unit.can_act()]


def move_to_closest_resource(unit, player, resource_tiles):
    # if the unit is a worker and we have space in cargo, lets find the nearest resource tile and try to mine it
    closest_dist = math.inf
    closest_resource_tile = None
    for resource_tile in resource_tiles:
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.COAL and not player.researched_coal(): continue
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.URANIUM and not player.researched_uranium(): continue
        dist = resource_tile.pos.distance_to(unit.pos)
        if dist < closest_dist:
            closest_dist = dist
            closest_resource_tile = resource_tile
    if closest_resource_tile is not None:
        return unit.move(unit.pos.direction_to(closest_resource_tile.pos))


def move_to_closest_city(unit, player):
    if len(player.cities) > 0:
        closest_dist = math.inf
        closest_city_tile = None
        for k, city in player.cities.items():
            for city_tile in city.citytiles:
                dist = city_tile.pos.distance_to(unit.pos)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_city_tile = city_tile
        if closest_city_tile is not None:
            move_dir = unit.pos.direction_to(closest_city_tile.pos)
            return unit.move(move_dir)
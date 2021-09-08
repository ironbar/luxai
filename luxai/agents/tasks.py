"""
Tasks for the units
"""
import random
from typing import List, Tuple

from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game_objects import Player, Unit, CityTile
from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game_map import Position
from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux import annotate
from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game_constants import GAME_CONSTANTS

from luxai.primitives import (
    is_position_in_list,
    find_closest_resource,
    find_closest_tile_to_unit,
    get_directions_to,
)
from luxai.game_info import GameInfo


class BaseTask():
    # TODO: priority property?
    def __init__(self):
        self.pos = None

    def is_done(self, unit: Unit) -> bool:
        """ Returns true when the task is already done """
        raise NotImplementedError()

    def get_actions(self, unit: Unit, game_info: GameInfo) -> Tuple[List[str], Position]:
        """
        Given the unit and a list of obstacle positions returns the actions that
        should be taken in order to do the task and also the future position of the unit

        It returns a list of actions because that allows to use annotations
        """
        return self._move_to_position(unit, game_info)

    def _move_to_position(self, unit: Unit, game_info: GameInfo) -> Tuple[List[str], Position]:
        if self.pos is None or self._keep_safe_a_home_at_night(unit, game_info):
            return [], unit.pos

        directions = get_directions_to(unit.pos, self.pos)
        random.shuffle(directions)
        for direction in directions:
            future_position = unit.pos.translate(direction, units=1)
            if is_position_in_list(future_position, game_info.obstacles):
                continue
            annotations = [
                annotate.line(unit.pos.x, unit.pos.y, self.pos.x, self.pos.y),
            ]
            return [unit.move(direction)] + annotations, future_position
        return [], unit.pos

    @staticmethod
    def _keep_safe_a_home_at_night(unit, game_info: GameInfo):
        if is_position_in_list(unit.pos, game_info.city_tile_positions):
            if unit.get_cargo_space_left() == GAME_CONSTANTS['PARAMETERS']['RESOURCE_CAPACITY']['WORKER']:
                if game_info.is_night:
                    return True
        return False


class GatherResourcesTask(BaseTask):
    def __init__(self, unit, game_info):
        super().__init__()
        self.update(unit, game_info)

    def update(self, unit, game_info):
        closest_resource_tile = find_closest_resource(unit, game_info.player, game_info.resource_tiles)
        if closest_resource_tile is None:
            self.pos = None
        else:
            self.pos = closest_resource_tile.pos

    def is_done(self, unit: Unit) -> bool:
        return not unit.get_cargo_space_left()


class GoToPositionTask(BaseTask):
    def is_done(self, unit: Unit) -> bool:
        return unit.pos.equals(self.pos)


class BuildCityTileTask(BaseTask):
    def __init__(self, unit, game_info):
        super().__init__()
        self.is_city_built = False
        self.update(unit, game_info)

    def update(self, unit, game_info):
        closest_empty_tile = find_closest_tile_to_unit(unit, game_info.empty_tiles)
        if closest_empty_tile is None:
            self.pos = None
        else:
            self.pos = closest_empty_tile.pos

    def is_done(self, unit: Unit) -> bool:
        return unit.pos.equals(self.pos) and self.is_city_built or unit.get_cargo_space_left()

    def get_actions(self, unit: Unit, game_info: GameInfo) -> Tuple[List[str], Position]:
        if not unit.pos.equals(self.pos):
            return self._move_to_position(unit, game_info)
        else:
            self.is_city_built = True
            return [unit.build_city()], unit.pos

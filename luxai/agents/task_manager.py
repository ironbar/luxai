"""
Agents that follow the task manager framework
"""
import random
from typing import List

from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game_constants import GAME_CONSTANTS

from luxai.agents.basic import BaseAgent
from luxai.agents.utils import (
    get_available_workers,
    get_non_available_workers,
    get_available_city_tiles,
    get_resource_tiles,
    get_empty_tiles,
    get_n_buildable_units,
    get_all_city_tiles,
    find_closest_tile_to_unit,
    find_closest_resource,
    find_closest_city_tile,
    move_to_closest_resource,
    move_to_closest_city_tile,
    is_position_in_list,
)
from luxai.agents.tasks import (
    GatherResourcesTask,
    GoToPositionTask,
    BuildCityTileTask,
)

class GameInfo():
    """
    Class to store all the relevant information of the game for taking decisions
    """
    def __init__(self):
        self.resource_tiles = None
        self.empty_tiles = None
        self.available_workers = None
        self.non_available_workers = None


class TaskManagerAgent(BaseAgent):
    """
    The philosophy of the agent is that it first assigns tasks to the agents, and later coordinates
    them based on the priority of their actions
    """
    def __init__(self):
        super().__init__()
        self.unit_id_to_task = {}
        self.game_info = GameInfo()
        self.player = None
        self.opponent = None

    def __call__(self, observation: dict, configuration: dict) -> List[str]:
        return self.task_manager(observation, configuration)

    def task_manager(self, observation: dict, configuration: dict) -> List[str]:
        self.gather_game_information(observation, configuration)
        self.assign_tasks_to_units()
        actions = self.coordinate_units_movement()
        actions.extend(self.manage_cities(self.player))
        return actions

    def gather_game_information(self, observation, configuration):
        """
        Updates the game_state and extracts information that later is used to take decisions
        """
        self._update_game_state(observation)
        self.player = self.game_state.players[observation.player]
        self.opponent = self.game_state.players[(observation.player + 1) % 2]
        self.game_info.resource_tiles = get_resource_tiles(self.game_state)
        self.game_info.empty_tiles = get_empty_tiles(self.game_state)
        random.shuffle(self.game_info.resource_tiles)
        random.shuffle(self.game_info.empty_tiles)

        self.game_info.available_workers = get_available_workers(self.player)
        random.shuffle(self.game_info.available_workers)
        self.game_info.non_available_workers = get_non_available_workers(self.player)

    def assign_tasks_to_units(self):
        """
        For the available units check if they already have a task, if that is the
        case verify if the task is already fullfilled and then assign a new task
        If the agent does not have a task assign one
        """
        for unit in self.game_info.available_workers:
            if unit.id in self.unit_id_to_task:
                task = self.unit_id_to_task[unit.id]
                if task.is_done(unit):
                    self.assign_new_task_to_unit(unit)
            else:
                self.assign_new_task_to_unit(unit)

    def assign_new_task_to_unit(self, unit):
        if not unit.get_cargo_space_left():
            # closest_city_tile = find_closest_city_tile(unit, self.player)
            # self.unit_id_to_task[unit.id] = GoToPositionTask(closest_city_tile.pos)
            closest_empty_tile = find_closest_tile_to_unit(unit, self.game_info.empty_tiles)
            self.unit_id_to_task[unit.id] = BuildCityTileTask(closest_empty_tile.pos)
        else:
            closest_resource_tile = find_closest_resource(unit, self.player, self.game_info.resource_tiles)
            self.unit_id_to_task[unit.id] = GatherResourcesTask(closest_resource_tile.pos)

    def coordinate_units_movement(self) -> List[str]:
        """
        For the available units coordinate the movements so they don't collide
        """
        assert all(unit.id in self.unit_id_to_task for unit in self.game_info.available_workers)
        actions = []
        obstacles = []
        for unit in self.game_info.available_workers:
            task = self.unit_id_to_task[unit.id]
            action, future_position = task.get_action(unit, obstacles)
            actions.append(action)
        return actions

    @staticmethod
    def manage_cities(player) -> List[str]:
        actions = []
        available_city_tiles = get_available_city_tiles(player)
        if available_city_tiles:
            n_buildable_units = get_n_buildable_units(player)
            for city_tile in available_city_tiles:
                if n_buildable_units:
                    n_buildable_units -= 1
                    actions.append(city_tile.build_worker())
                elif player.research_points < GAME_CONSTANTS['PARAMETERS']['RESEARCH_REQUIREMENTS']['URANIUM']:
                    actions.append(city_tile.research())
        return actions



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

class GameInfo():
    """
    Class to store all the relevant information of the game for taking decisions
    """
    def __init__(self):
        self.resource_tiles = None
        self.empty_tiles = None


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
        resource_tiles = get_resource_tiles(self.game_state)
        empty_tiles = get_empty_tiles(self.game_state)
        random.shuffle(empty_tiles)
        random.shuffle(resource_tiles)
        self.game_info.resource_tiles = resource_tiles
        self.game_info.empty_tiles = empty_tiles

    def assign_tasks_to_units(self):
        """
        For the available units check if they already have a task, if that is the
        case verify if the task is already fullfilled and then assign a new task
        If the agent does not have a task assign one
        """
        pass

    def coordinate_units_movement(self) -> List[str]:
        """
        For the available units coordinate the movements so they don't collide
        """
        return []

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

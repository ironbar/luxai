"""
Agents that follow the task manager framework
"""
import random
from typing import List

from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux import annotate

from luxai.primitives import is_position_in_list
from luxai.agents.tasks import (
    GatherResourcesTask,
    BuildCityTileTask
)
from luxai.game_info import GameInfo

class TaskManagerAgent():
    """
    The philosophy of the agent is that it first assigns tasks to the agents, and later coordinates
    them based on the priority of their actions
    """
    def __init__(self):
        self.unit_id_to_task = {}
        self.game_info = GameInfo()
        self.actions = []

    def __call__(self, observation: dict, configuration: dict) -> List[str]:
        return self.task_manager(observation, configuration)

    def task_manager(self, observation: dict, configuration: dict) -> List[str]:
        self.actions = []
        self.gather_game_information(observation, configuration)
        self.assign_tasks_to_units()
        self.coordinate_units_movement()
        self.manage_cities()
        return self.actions

    def gather_game_information(self, observation, configuration):
        """
        Updates the game_state and extracts information that later is used to take decisions
        """
        self.game_info.update(observation, configuration)
        if self.game_info.is_night:
            self.actions.append(annotate.sidetext('Night'))

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
                    task.update(unit, self.game_info)
            else:
                self.assign_new_task_to_unit(unit)

    def assign_new_task_to_unit(self, unit):
        if not unit.get_cargo_space_left():
            # closest_city_tile = find_closest_city_tile(unit, self.player)
            # self.unit_id_to_task[unit.id] = GoToPositionTask(closest_city_tile.pos)
            self.unit_id_to_task[unit.id] = BuildCityTileTask(unit, self.game_info)
        else:
            self.unit_id_to_task[unit.id] = GatherResourcesTask(unit, self.game_info)

    def coordinate_units_movement(self):
        """
        For the available units coordinate the movements so they don't collide
        """
        assert all(unit.id in self.unit_id_to_task for unit in self.game_info.available_workers)
        actions = []
        for unit in self.game_info.available_workers:
            task = self.unit_id_to_task[unit.id]
            unit_actions, future_position = task.get_actions(unit, self.game_info)
            actions.extend(unit_actions)
            if not is_position_in_list(future_position, self.game_info.city_tile_positions):
                self.game_info.obstacles.append(future_position)
        # actions += [annotate.x(position.x, position.y) for position in obstacles]
        self.actions.extend(actions)

    def manage_cities(self):
        for city_tile in self.game_info.available_city_tiles:
            if self.game_info.n_buildable_units:
                self.game_info.n_buildable_units -= 1
                self.actions.append(city_tile.build_worker())
            elif self.game_info.research_points_to_uranium:
                self.game_info.research_points_to_uranium -= 1
                self.actions.append(city_tile.research())

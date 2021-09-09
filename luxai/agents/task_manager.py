"""
Agents that follow the task manager framework
"""
import random
from typing import List

from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux import annotate

from luxai.primitives import is_position_in_list
from luxai.agents.tasks import (
    GatherResourcesTask,
    BuildCityTileTask,
    GoToClosestCity
)
from luxai.game_info import GameInfo

class TaskManagerAgent():
    """
    The philosophy of the agent is that it first assigns tasks to the agents, and later coordinates
    them based on the priority of their actions
    """
    def __init__(self, build_new_city_tile_probability):
        self.unit_id_to_task = {}
        self.game_info = GameInfo()
        self.actions = []
        self.build_new_city_tile_probability = build_new_city_tile_probability

    def __call__(self, observation: dict, configuration: dict) -> List[str]:
        return self.task_manager(observation, configuration)

    def task_manager(self, observation: dict, configuration: dict) -> List[str]:
        self.actions = []
        self.game_info.update(observation, configuration)
        self.assign_tasks_to_units()
        self.coordinate_units_movement()
        self.manage_cities()
        self.annotations()
        return self.actions

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
            build_new_city_tile = random.uniform(0, 1) < self.build_new_city_tile_probability
            if build_new_city_tile:
                self.unit_id_to_task[unit.id] = BuildCityTileTask(unit, self.game_info)
            else:
                self.unit_id_to_task[unit.id] = GoToClosestCity(unit, self.game_info)
        else:
            self.unit_id_to_task[unit.id] = GatherResourcesTask(unit, self.game_info)

    def coordinate_units_movement(self):
        """
        For the available units coordinate the movements so they don't collide
        """
        assert all(unit.id in self.unit_id_to_task for unit in self.game_info.available_workers)
        for unit in self.game_info.available_workers:
            task = self.unit_id_to_task[unit.id]
            unit_actions, future_position = task.get_actions(unit, self.game_info)
            self.actions.extend(unit_actions)
            if not is_position_in_list(future_position, self.game_info.city_tile_positions):
                self.game_info.obstacles.append(future_position)

    def manage_cities(self):
        for city_tile in self.game_info.available_city_tiles:
            if self.game_info.n_buildable_units > 0:
                self.game_info.n_buildable_units -= 1
                self.actions.append(city_tile.build_worker())
            elif self.game_info.research_points_to_uranium > 0:
                self.game_info.research_points_to_uranium -= 1
                self.actions.append(city_tile.research())

    def annotations(self):
        # self.actions.append(annotate.sidetext('Research points to uranium: %i' % self.game_info.research_points_to_uranium))
        pass
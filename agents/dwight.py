"""
The name dwight comes because it is a task manager but not as good to be called michael scott
"""

"""
Agents that follow the task manager framework
"""
import math
import random
from typing import List, Tuple

from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux import annotate
from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game import Game
from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game_constants import GAME_CONSTANTS
from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game_objects import Player, Unit, CityTile
from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game_map import Position
from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux import annotate
from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game_constants import GAME_CONSTANTS
from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.constants import Constants
from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game import Game
from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game_objects import Player, Unit, CityTile
from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game_map import Cell, Position


class GameInfo():
    """
    Class to store all the relevant information of the game for taking decisions
    """
    def __init__(self):
        self.game_state = None
        self.player = None
        self.opponent = None
        self.resource_tiles = None
        self.empty_tiles = None
        self.available_workers = None
        self.non_available_workers = None
        self.city_tiles = None
        self.city_tile_positions = None
        self.opponent_city_tile_positions = None
        self.obstacles = None
        self.is_night = None
        self.available_city_tiles = None
        self.n_buildable_units = None
        self.research_points_to_uranium = None

    def update(self, observation, configuration):
        """
        Updates the game_state and extracts information that later is used to take decisions
        """
        self._update_game_state(observation)
        self.player = self.game_state.players[observation.player]
        self.opponent = self.game_state.players[(observation.player + 1) % 2]
        self.resource_tiles = get_resource_tiles(self.game_state)
        self.empty_tiles = get_empty_tiles(self.game_state)
        random.shuffle(self.resource_tiles)
        random.shuffle(self.empty_tiles)

        self.available_workers = get_available_workers(self.player)
        random.shuffle(self.available_workers)
        self.non_available_workers = get_non_available_workers(self.player)

        self.city_tiles = get_all_city_tiles(self.player)
        self.city_tile_positions = [city_tile.pos for city_tile in self.city_tiles]
        self.opponent_city_tile_positions = [city_tile.pos for city_tile in get_all_city_tiles(self.opponent)]
        self.available_city_tiles = get_available_city_tiles(self.player)
        self.n_buildable_units = get_n_buildable_units(self.player)

        self.obstacles = [unit.pos for unit in self.non_available_workers if \
                          not is_position_in_list(unit.pos, self.city_tile_positions)]
        self.obstacles += self.opponent_city_tile_positions

        self.is_night = observation["step"] % 40 >= 30

        self.research_points_to_uranium = GAME_CONSTANTS['PARAMETERS']['RESEARCH_REQUIREMENTS']['URANIUM'] - self.player.research_points

    def _update_game_state(self, observation):
        if observation["step"] == 0:
            self.game_state = Game()
            self.game_state._initialize(observation["updates"])
            self.game_state._update(observation["updates"][2:])
            self.game_state.id = observation.player
        else:
            self.game_state._update(observation["updates"])


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
        return not unit.get_cargo_space_left() or self.pos is None


class GoToClosestCity(BaseTask):
    def __init__(self, unit, game_info):
        super().__init__()
        self.update(unit, game_info)

    def update(self, unit, game_info):
        closest_tile = find_closest_tile_to_unit(unit, game_info.city_tiles)
        if closest_tile is None:
            self.pos = None
        else:
            self.pos = closest_tile.pos

    def is_done(self, unit: Unit) -> bool:
        return self.pos is None or unit.pos.equals(self.pos)


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
        return self.pos is None or unit.pos.equals(self.pos) and self.is_city_built or unit.get_cargo_space_left()

    def get_actions(self, unit: Unit, game_info: GameInfo) -> Tuple[List[str], Position]:
        if not unit.pos.equals(self.pos):
            return self._move_to_position(unit, game_info)
        else:
            self.is_city_built = True
            return [unit.build_city()], unit.pos


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
        for unit in self.game_info.available_workers + self.game_info.non_available_workers:
            task = self.unit_id_to_task[unit.id]
            if task.pos is not None:
                self.actions.append(annotate.line(unit.pos.x, unit.pos.y, task.pos.x, task.pos.y))
        pass

global_agent = TaskManagerAgent(0.9)

def agent(observation, configuration):
    return global_agent(observation, configuration)
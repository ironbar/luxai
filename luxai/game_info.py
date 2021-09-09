"""
"""
import random

from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game import Game
from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game_constants import GAME_CONSTANTS

from luxai.primitives import (
    get_available_workers,
    get_non_available_workers,
    get_available_city_tiles,
    get_resource_tiles,
    get_empty_tiles,
    get_n_buildable_units,
    get_all_city_tiles,
    is_position_in_list,
)


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

        self.is_night = observation.step % 40 >= 30

        self.research_points_to_uranium = GAME_CONSTANTS['PARAMETERS']['RESEARCH_REQUIREMENTS']['URANIUM'] - self.player.research_points

    def _update_game_state(self, observation):
        if observation["step"] == 0:
            self.game_state = Game()
            self.game_state._initialize(observation["updates"])
            self.game_state._update(observation["updates"][2:])
            self.game_state.id = observation.player
        else:
            self.game_state._update(observation["updates"])
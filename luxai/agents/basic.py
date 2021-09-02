"""
Basic handmade agents
"""
import random

from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game import Game
from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game_objects import Player
from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux import annotate

from luxai.agents.utils import (
    get_available_workers,
    get_available_city_tiles,
    get_resource_tiles,
    get_empty_tiles,
    get_n_buildable_units,
    find_closest_tile_to_unit,
    move_to_closest_resource,
    move_to_closest_city_tile,
)

class BaseAgent():
    def __init__(self):
        self.game_state = None

    def _update_game_state(self, observation):
        if observation["step"] == 0:
            self.game_state = Game()
            self.game_state._initialize(observation["updates"])
            self.game_state._update(observation["updates"][2:])
            self.game_state.id = observation.player
        else:
            self.game_state._update(observation["updates"])

    def __call__(self, observation: dict, configuration: dict) -> list[str]:
        self._update_game_state(observation)
        raise NotImplementedError('You have to implement this function')


class SimpleAgent(BaseAgent):
    """ This agent simply replicates the simple_agent provided by kaggle """
    def __init__(self):
        super().__init__()

    def __call__(self, observation: dict, configuration: dict) -> list[str]:
        self._update_game_state(observation)
        player = self.game_state.players[observation.player]
        resource_tiles = get_resource_tiles(self.game_state)
        actions = self.create_simple_actions_for_workers(player, resource_tiles)
        actions = [action for action in actions if action is not None]
        return actions

    @staticmethod
    def create_simple_actions_for_workers(player, resource_tiles) -> list[str]:
        actions = []
        for unit in get_available_workers(player):
            if unit.get_cargo_space_left() > 0:
                actions.append(move_to_closest_resource(unit, player, resource_tiles))
            else:
                actions.append(move_to_closest_city_tile(unit, player))
        return actions


class ResearchAgent(SimpleAgent):
    """ This agent extends the simple agent by doing research with the city """
    def __init__(self):
        super().__init__()

    def __call__(self, observation: dict, configuration: dict) -> list[str]:
        self._update_game_state(observation)
        player = self.game_state.players[observation.player]
        resource_tiles = get_resource_tiles(self.game_state)
        actions = self.create_simple_actions_for_workers(player, resource_tiles)
        actions.extend(self.make_city_tiles_research_whenever_possible(player))
        actions = [action for action in actions if action is not None]
        return actions

    @staticmethod
    def make_city_tiles_research_whenever_possible(player: Player) -> list[str]:
        actions = []
        for city_tile in get_available_city_tiles(player):
            actions.append(city_tile.research())
        return actions


class BuildWorkerOrResearchAgent(SimpleAgent):
    """
    This agent extends the simple agent by building agents whenever possible or researching
    otherwise with the city tiles
    """
    def __init__(self):
        super().__init__()

    def __call__(self, observation: dict, configuration: dict) -> list[str]:
        self._update_game_state(observation)
        player = self.game_state.players[observation.player]
        resource_tiles = get_resource_tiles(self.game_state)
        actions = self.create_simple_actions_for_workers(player, resource_tiles)
        actions.extend(self.manage_city_tiles(player))
        actions = [action for action in actions if action is not None]
        return actions

    @staticmethod
    def manage_city_tiles(player: Player) -> list[str]:
        actions = []
        available_city_tiles = get_available_city_tiles(player)
        if available_city_tiles:
            n_buildable_units = get_n_buildable_units(player)
            for city_tile in available_city_tiles:
                if n_buildable_units:
                    n_buildable_units -= 1
                    actions.append(city_tile.build_worker())
                else:
                    actions.append(city_tile.research())
        return actions


class NaiveViralAgent(BuildWorkerOrResearchAgent):
    """
    This agent will build new city tiles to grow as fast as possible
    """
    def __init__(self, build_new_city_tile_probability):
        super().__init__()
        self.build_new_city_tile_probability = build_new_city_tile_probability

    def __call__(self, observation: dict, configuration: dict) -> list[str]:
        self._update_game_state(observation)
        player = self.game_state.players[observation.player]
        resource_tiles = get_resource_tiles(self.game_state)
        empty_tiles = get_empty_tiles(self.game_state)

        actions = self.create_viral_actions_for_workers(player, resource_tiles, empty_tiles)
        actions.extend(self.manage_city_tiles(player))
        actions = [action for action in actions if action is not None]
        return actions

    def create_viral_actions_for_workers(self, player, resource_tiles, empty_tiles) -> list[str]:
        actions = []
        for unit in get_available_workers(player):
            if unit.get_cargo_space_left() > 0:
                actions.append(move_to_closest_resource(unit, player, resource_tiles))
            else:
                build_new_city_tile = random.uniform(0, 1) < self.build_new_city_tile_probability
                if build_new_city_tile:
                    closest_empty_tile = find_closest_tile_to_unit(unit, empty_tiles)
                    if closest_empty_tile is not None:
                        #actions.append(annotate.sidetext('Closest empty tile: %s' % str(closest_empty_tile.pos)))
                        if unit.pos.equals(closest_empty_tile.pos):
                            actions.append(unit.build_city())
                        else:
                            actions.append(unit.move(unit.pos.direction_to(closest_empty_tile.pos)))
                else:
                    actions.append(move_to_closest_city_tile(unit, player))
        return actions


class NaiveRandomViralAgent(NaiveViralAgent):
    """
    This agent extends NaiveViralAgent by randomly shuffling the tiles
    This make the agent much more stronger because it is capable of avoiding some blockings
    """
    def __init__(self, build_new_city_tile_probability):
        super().__init__(build_new_city_tile_probability)

    def __call__(self, observation: dict, configuration: dict) -> list[str]:
        self._update_game_state(observation)
        player = self.game_state.players[observation.player]
        resource_tiles = get_resource_tiles(self.game_state)
        empty_tiles = get_empty_tiles(self.game_state)
        random.shuffle(empty_tiles)
        random.shuffle(resource_tiles)

        actions = self.create_viral_actions_for_workers(player, resource_tiles, empty_tiles)
        actions.extend(self.manage_city_tiles(player))
        actions = [action for action in actions if action is not None]
        return actions

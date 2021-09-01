"""
Basic handmade agents
"""
from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game import Game

from luxai.agents.utils import (
    get_available_workers,
    get_resource_tiles,
    move_to_closest_resource,
    move_to_closest_city_tile
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

    def __call__(self, observation, configuration):
        self._update_game_state(observation)
        raise NotImplementedError('You have to implement this function')


class SimpleAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    def __call__(self, observation, configuration):
        self._update_game_state(observation)

        ### AI Code goes down here! ###
        player = self.game_state.players[observation.player]
        #opponent = game_state.players[(observation.player + 1) % 2]
        resource_tiles = get_resource_tiles(self.game_state)
        actions = []
        # we iterate over all our units and do something with them
        for unit in get_available_workers(player):
            if unit.get_cargo_space_left() > 0:
                actions.append(move_to_closest_resource(unit, player, resource_tiles))
            else:
                actions.append(move_to_closest_city_tile(unit, player))

        actions = [action for action in actions if action is not None]
        return actions

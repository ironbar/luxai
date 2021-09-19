import sys
import os
import argparse
import importlib
from typing import List
from kaggle_environments import make
import cv2

from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game import Game

from luxai.utils import render_game_in_html, set_random_seed, update_game_state
from luxai.render import render_game_state, get_captions

DEFAULT_AGENT = '/mnt/hdd0/MEGA/AI/22 Kaggle/luxai/agents/working_title/agent.py'


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    set_random_seed(args.random_seed)
    game_conf = {'width': args.size, 'height': args.size, 'seed': args.map_seed,
                 'actTimeout': int(1e6), 'runTimeout': int(1e6),
                 'episodeSteps': 361, 'annotations':True}
    env = make("lux_ai_2021", debug=True, configuration=game_conf)
    game_inteface = GameInterface(args.player0)
    game_info = env.run([game_inteface, args.player1])
    # render_game_in_html(env)


class GameInterface():
    def __init__(self, agent_script, window_name='luxai game player'):
        self.game_state = Game()
        sys.path.append(os.path.dirname(agent_script))
        self.agent = importlib.import_module(os.path.splitext(os.path.basename(agent_script))[0])
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        self.window_name = window_name

    def __call__(self, observation: dict, configuration: dict) -> List[str]:
        update_game_state(self.game_state, observation)
        actions = self.agent.agent(observation, configuration)
        self.render_step()
        return actions

    def render_step(self, top_border=128):
        render = render_game_state(self.game_state)
        if top_border:
            render = cv2.copyMakeBorder(render, top_border, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        caption = get_captions(self.game_state)
        caption = 'Turn %i/360\n%s' % (self.game_state.turn, caption)
        cv2.imshow(self.window_name, render)
        cv2.displayOverlay(self.window_name, caption)
        cv2.waitKey(1)

    def __del__(self):
        cv2.destroyAllWindows()
        cv2.destroyAllWindows()
        cv2.destroyAllWindows()
        cv2.destroyAllWindows()


def parse_args(args):
    epilog = """
    """
    description = """
    Interface for playing luxai game
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('--player0', help='Path to python script for player0', default=DEFAULT_AGENT)
    parser.add_argument('--player1', help='Path to python script for player1', default=DEFAULT_AGENT)
    parser.add_argument('--map_seed', help='Seed to generate the map', default=0, type=int)
    parser.add_argument('--size', help='Size of the map', default=12, type=int)
    parser.add_argument('--random_seed', help='Seed for the agents', default=7, type=int)
    # TODO: add the option to use checkpoints
    return parser.parse_args(args)


if __name__ == '__main__':
    main()

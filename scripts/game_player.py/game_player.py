import sys
import os
import argparse
import importlib
from typing import List
from kaggle_environments import make
import cv2

from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game import Game
from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game_objects import Unit, CityTile

from luxai.utils import render_game_in_html, set_random_seed, update_game_state
from luxai.render import render_game_state, get_captions, add_actions_to_render, show_focus_on_active_unit
from luxai.primitives import get_available_city_tiles, get_available_units

DEFAULT_AGENT = '/mnt/hdd0/MEGA/AI/22 Kaggle/luxai/agents/working_title/agent.py'


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    set_random_seed(args.random_seed)
    game_conf = {'width': args.size, 'height': args.size, 'seed': args.map_seed,
                 'actTimeout': int(1e6), 'runTimeout': int(1e6),
                 'episodeSteps': args.episode_steps, 'annotations':True}
    env = make("lux_ai_2021", debug=True, configuration=game_conf)
    game_inteface = GameInterface(args.player0)
    game_info = env.run([game_inteface, args.player1])
    # TODO: better naming for saving the file
    with open('delete.json', 'w') as f:
        f.write(env.render(mode='json'))


class GameInterface():
    def __init__(self, agent_script, window_name='luxai game player'):
        self.game_state = Game()
        sys.path.append(os.path.dirname(agent_script))
        self.agent = importlib.import_module(os.path.splitext(os.path.basename(agent_script))[0])
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        self.window_name = window_name
        self.game_interface_is_on = True
        # TODO: add a tqdm to show game progress

    def __call__(self, observation: dict, configuration: dict) -> List[str]:
        update_game_state(self.game_state, observation)
        render, caption = self.render_step()
        return self.game_interface(render, caption, observation, configuration)

    def render_step(self):
        render = render_game_state(self.game_state)
        caption = get_captions(self.game_state)
        caption = 'Turn %i/360\n%s' % (self.game_state.turn, caption)
        return render, caption

    def game_interface(self, render, caption, observation, configuration):
        actions = self.agent.agent(observation, configuration)
        actions = remove_annotations(actions)
        available_units = get_available_units_and_cities(self.game_state.players[0])
        unit_idx = 0
        while self.game_interface_is_on and available_units:
            unit = available_units[unit_idx]
            updated_render = add_actions_to_render(render, actions, self.game_state)
            show_focus_on_active_unit(updated_render, unit)
            updated_caption = caption + '\nAvailable units: %i Actions: %s' % (len(available_units), str(actions))
            key = self.display_render(updated_render, updated_caption)
            # print(key)
            if key == 27: # ESC
                print('Turning off game interface, game will continue automatically until the end')
                self.game_interface_is_on = False
            if key == 13: # ENTER:
                pass
            if key == 32: # SPACE:
                break
            if key == 83: # ->:
                pass
            # Change between objects
            if key == ord('4'):
                unit_idx = (unit_idx - 1)%len(available_units)
            if key == ord('6'):
                unit_idx = (unit_idx + 1)%len(available_units)
            if isinstance(unit, Unit):
                for letter, direction in [('w', 'n'), ('s', 's'), ('a', 'w'), ('d', 'e'), ('c', 'c')]:
                    if key == ord(letter):
                        new_action = 'm %s %s' % (unit.id, direction)
                        update_unit_action(actions, unit, new_action)
                if key == ord('b'):
                    new_action = 'bcity %s' % (unit.id)
                    update_unit_action(actions, unit, new_action)
            if isinstance(unit, CityTile):
                if key == ord('r'):
                    update_citytile_action(actions, unit, 'r')
                if key == ord('w'):
                    update_citytile_action(actions, unit, 'bw')
                if key == ord('c'):
                    update_citytile_action(actions, unit, 'bc')

        return actions

    def display_render(self, render, caption, top_border=128):
        if top_border:
            render = cv2.copyMakeBorder(render, top_border, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        cv2.imshow(self.window_name, render[:, :, [2, 1, 0]]) # opencv uses bgr convention while the images are rgb
        cv2.displayOverlay(self.window_name, caption)
        key = cv2.waitKey(500) & 0xFF
        return key

    def __del__(self):
        cv2.destroyAllWindows()
        cv2.destroyAllWindows()
        cv2.destroyAllWindows()
        cv2.destroyAllWindows()


def remove_annotations(actions):
    return list(filter(is_action, actions))
    return actions


def is_action(action):
    return any(action.startswith(start) for start in ['m ', 't ', 'bcity ', 'p ', 'r ', 'bw ', 'bc '])


def get_available_units_and_cities(player):
    return get_available_units(player) + get_available_city_tiles(player)


def update_unit_action(actions, unit, new_action):
    for idx, action in enumerate(actions):
        if action.split(' ')[1] == unit.id:
            actions[idx] = new_action


def update_citytile_action(actions, citytile, command):
    end = ' %i %i' % (citytile.pos.x, citytile.pos.y)
    for idx, action in enumerate(actions):
        if action.endswith(end):
            actions[idx] = command + end


def parse_args(args):
    epilog = """
    python scripts/game_player.py/game_player.py
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
    parser.add_argument('--episode_steps', help='Number of steps of the game', default=361, type=int)
    # TODO: add the option to use checkpoints
    return parser.parse_args(args)


if __name__ == '__main__':
    main()

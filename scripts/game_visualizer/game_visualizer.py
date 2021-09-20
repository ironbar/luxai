"""
Game visualizer

TODO:

- [x] Render game on the fly instead of rendering the whole game at the start of the game
"""
import sys
import argparse
import json
from tqdm import tqdm
import cv2
from functools import partial

from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game import Game

from luxai.render import render_game_state, get_captions
from luxai.utils import update_game_state

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    game_visualizer = GameVisualizer(args.game_path)
    game_visualizer.run()

    # with open(args.game_path, 'r') as json_file:
    #     game_info = json.load(json_file)['steps'][:20]

    # renders, captions = render_whole_game(game_info)
    # visualize_game(renders, captions)


class GameVisualizer():
    """
    Class that stores renders and captions for visualizing the game
    and avoiding using global variables
    """
    def __init__(self, game_path):
        with open(game_path, 'r') as json_file:
            self.game_info = json.load(json_file)['steps']
        self.renders, self.captions = dict(), dict()

    def run(self):
        self.visualize_game()

    def _get_step_render_info(self, epoch, top_border=128):
        if epoch not in self.renders:
            game_state = get_game_state_for_epoch(self.game_info, epoch)
            render = render_game_state(game_state)
            if top_border:
                render = cv2.copyMakeBorder(render, top_border, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            self.renders[epoch] = render
            self.captions[epoch] = get_captions(game_state)
        return self.renders[epoch], self.captions[epoch]

    def visualize_game(self, window_name='render'):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        callback = partial(self.update_window, window_name=window_name)
        cv2.createTrackbar('step', window_name, 0, len(self.game_info)-1, callback)
        callback(0)
        while 1:
            k = cv2.waitKey(1) & 0xFF
            if k == 27: # ESC
                break
        cv2.destroyAllWindows()
        cv2.destroyAllWindows()
        cv2.destroyAllWindows()
        cv2.destroyAllWindows()

    def update_window(self, epoch, window_name):
        render, caption = self._get_step_render_info(epoch)
        cv2.imshow(window_name, render[:, :, [2, 1, 0]]) # opencv uses bgr convention while the images are rgb
        cv2.displayOverlay(window_name, caption)


def get_game_state_for_epoch(game_info, epoch):
    """ Returns the game state for the desired epoch [0-360] """
    game_state = Game()
    for step_info in game_info[:epoch+1]:
        update_game_state(game_state, step_info[0]['observation'])
    return game_state


def parse_args(args):
    epilog = """
    python scripts/game_visualizer/game_visualizer.py notebooks/sample_game.json
    """
    description = """
    Simple demonstration of game visualization using opencv
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('game_path', help='Path to json file with game information')
    return parser.parse_args(args)


if __name__ == '__main__':
    main()

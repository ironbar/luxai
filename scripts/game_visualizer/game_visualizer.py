import sys
import argparse
import json
from tqdm import tqdm
import cv2
from functools import partial

from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game import Game

from luxai.render import render_game_state

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    with open(args.game_path, 'r') as json_file:
        game_info = json.load(json_file)['steps'][:20]

    renders, captions = render_whole_game(game_info)
    visualize_game(renders, captions)


def render_whole_game(game_info):
    renders, captions = [], []
    game_state = Game()
    for step_info in tqdm(game_info, desc='rendering game'):
        _update_game_state(game_state, step_info)
        renders.append(render_game_state(game_state))
        captions.append('')
    return renders, captions


def _update_game_state(game_state, step_info):
    observation = step_info[0]['observation']
    if observation["step"] == 0:
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation['player']
    else:
        game_state._update(observation["updates"])


def visualize_game(renders, captions, window_name='render'):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    callback = partial(update_window, window_name=window_name, renders=renders, captions=captions)
    cv2.createTrackbar('step', window_name, 0, len(renders)-1, callback)
    callback(0)
    while 1:
        k = cv2.waitKey(1) & 0xFF
        if k == 27: # ESC
            break
    cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    cv2.destroyAllWindows()


def update_window(step_idx, window_name, renders, captions):
    cv2.imshow(window_name, renders[step_idx][:, :, [2, 1, 0]])
    cv2.displayOverlay(window_name, captions[step_idx])


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

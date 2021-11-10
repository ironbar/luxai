"""
Functions commonly used in the challenge
"""
import os
import random
import time
import tempfile
import logging
import numpy as np
from tqdm import tqdm


def get_timestamp():
    time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    return time_stamp


def render_game_in_html(env, filepath=None):
    """
    Saves the game on html format and opens it with google chrome
    This allows to visualize the game using the full tab
    If a filepath is not given a temporal one will be used
    """
    if filepath is None:
        filepath = tempfile.NamedTemporaryFile(delete=False, suffix='.html').name
    with open(filepath, 'w') as f:
        f.write(env.render(mode='html'))
    os.system('google-chrome "%s"' % os.path.realpath(filepath))


def create_temporal_python_file(text: str) -> str:
    """
    Creates a temporal python file with the provided text and returns the path to the file
    """
    filepath = tempfile.NamedTemporaryFile(delete=False, suffix='.py').name
    with open(filepath, 'w') as f:
        f.write(text)
    return filepath


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def update_game_state(game_state, observation):
    if observation["step"] == 0 or not hasattr(game_state, 'map_width'):
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation['player']
    else:
        game_state._update(observation["updates"])
    game_state.turn = observation['step']


def monitor_submits_progress(submits, desc=None):
    """ Shows a progress bar representing the jobs done """
    progress_bar = tqdm(total=len(submits), desc=desc)
    progress = 0
    while 1:
        time.sleep(1)
        current_progress = np.sum([submit.done() for submit in submits])
        if current_progress > progress:
            progress_bar.update(current_progress - progress)
            progress = current_progress
        if progress == len(submits):
            break
    time.sleep(0.1)
    progress_bar.close()

def configure_logging(level=logging.DEBUG):
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
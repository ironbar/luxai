"""
Functions commonly used in the challenge
"""
import os
import random
import time
import tempfile
import numpy as np


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
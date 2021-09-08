import time
import random
import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from concurrent.futures import ProcessPoolExecutor
from kaggle_environments import evaluate

from luxai.definitions import BOARD_SIZES
from luxai.utils import set_random_seed


def play_matches_in_parallel(agents, max_workers=20, n_matches=100,
                             running_on_notebook=True):
    """
    Plays n_matches in parallel using ProcessPoolExecutor
    There might be blocking if a single game is played previously on the notebook

    Parameters
    -----------
    agents : list
        List with the names or paths to the agents
    """
    assert len(agents) == 2
    assert all(isinstance(agent, str) for agent in agents)

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        submits = []
        for submit_idx in range(n_matches):
            size = BOARD_SIZES[submit_idx % len(BOARD_SIZES)]
            configuration = {'width': size, 'height': size, 'seed': submit_idx}
            submits.append(pool.submit(play_game, agents=agents, configuration=configuration))
        monitor_progress(submits, running_on_notebook)
        matches_results = [submit.result()[0] for submit in submits]
    return matches_results


def play_game(agents, configuration):
    set_random_seed(configuration['seed'])
    return evaluate(environment="lux_ai_2021", agents=agents, configuration=configuration, num_episodes=1)


def monitor_progress(submits, running_on_notebook):
    if running_on_notebook:
        progress_bar = tqdm_notebook(total=len(submits))
    else:
        progress_bar = tqdm(total=len(submits))
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


def compute_result_ratios(matches_results):
    """
    Given the matches results computes win, tie and loss ratio from the perspective of the first
    agent
    """
    matches_results = np.array(matches_results)
    win_ratio = np.mean(matches_results[:, 0] > matches_results[:, 1])
    tie_ratio = np.mean(matches_results[:, 0] == matches_results[:, 1])
    loss_ratio = np.mean(matches_results[:, 0] < matches_results[:, 1])
    return win_ratio, tie_ratio, loss_ratio

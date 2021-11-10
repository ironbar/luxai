import sys
import argparse
import os
import time
import pandas as pd
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from luxai.data import load_match_from_json, save_match_to_npz
from luxai.utils import monitor_submits_progress


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    df = pd.read_csv(args.dataframe_path)
    episode_id_and_player_pairs = list(zip(df.EpisodeId, df.Index))
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        submits = []
        for episode_id, player in tqdm(episode_id_and_player_pairs, desc='creating jobs'):
            submits.append(pool.submit(preprocess_match, episode_id, player, args.matches_json_dir,
                                       args.matches_cache_npz_dir))
        monitor_submits_progress(submits)


def preprocess_match(episode_id, player, matches_json_dir, matches_cache_npz_dir):
    npz_filepath = os.path.join(matches_cache_npz_dir, '%i_%i.npz' % (episode_id, player))
    if os.path.exists(npz_filepath):
        return

    json_filepath = os.path.join(matches_json_dir, '%i.json' % episode_id)
    match = load_match_from_json(json_filepath, player)
    save_match_to_npz(npz_filepath, match)


def parse_args(args):
    epilog = """
    python json_to_npz.py /home/gbarbadillo/luxai_ssd/agent_selection_20211102_teams.csv /home/gbarbadillo/luxai_ssd/matches_20211014/matches_json /home/gbarbadillo/luxai_ssd/matches_20211014/matches_npz
    """
    description = """
    Takes json files and transforms them to npz
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('dataframe_path', help='')
    parser.add_argument('matches_json_dir', help='')
    parser.add_argument('matches_cache_npz_dir', help='')
    parser.add_argument('--max_workers', default=20, type=int, help='')
    return parser.parse_args(args)


if __name__ == '__main__':
    main()

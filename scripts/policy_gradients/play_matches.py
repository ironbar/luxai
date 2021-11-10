import sys
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from kaggle_environments import make

from luxai.utils import monitor_submits_progress

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    os.makedirs(args.output_folder, exist_ok=True)

    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        submits = []
        for match_idx in tqdm(range(args.n_matches), desc='creating jobs'):
            submits.append(pool.submit(play_and_save_match, args.agent1, args.agent2,
                                       match_idx, args.output_folder))
        monitor_submits_progress(submits, desc='playing matches')


def play_and_save_match(agent1, agent2, match_idx, output_folder):
    env = make("lux_ai_2021")
    env.run([agent1, agent2])
    filepath = os.path.join(output_folder, '%04d.json' % match_idx)
    with open(filepath, 'w') as f:
        f.write(env.render(mode='json'))


def parse_args(args):
    epilog = """
    python play_matches.py temp simple_agent "/mnt/hdd0/MEGA/AI/22 Kaggle/luxai/agents/superfocus_64/agent.py" --n_matches 100
    """
    description = """
    Plays matches in parallel and saves them to json file
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('output_folder', help='Path to folder where json files will be saved')
    parser.add_argument('agent1', help='Path to agent.py')
    parser.add_argument('agent2', help='Path to agent.py')
    parser.add_argument('--n_matches', type=int, default=100, help='Number of matches to play')
    parser.add_argument('--n_workers', type=int, default=20, help='Number of workers to run the matches')
    return parser.parse_args(args)


if __name__ == '__main__':
    main()

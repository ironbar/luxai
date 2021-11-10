import os
import sys
import argparse
import yaml
import json
import logging
import glob
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from luxai.data import load_match_from_json, combine_data_for_training
from luxai.utils import configure_logging

sys.path.append('../train_imitation_learning')
from train import create_model
from train_data_augmentation import train_generator

logger = logging.getLogger(__name__)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    train(args.config_path)


def train(config_path):
    configure_logging(logging.INFO)
    with open(config_path) as f:
        train_conf = yaml.safe_load(f)
    output_folder = os.path.dirname(os.path.realpath(config_path))
    tensorboard_writer = tf.summary.create_file_writer(os.path.join(output_folder, 'logs'))

    model = create_model(train_conf['model_params'])
    for epoch in range(train_conf['max_epochs']):
        print()
        logger.info('Start epoch %i' % epoch)
        play_matches(train_conf['play_matches'])
        logger.info('Loading train data')
        train_data, results = load_train_data(train_conf['play_matches']['output_folder'])
        log_matches_results(results, epoch, tensorboard_writer)
        logger.info('Fitting the model')
        model.fit(x=train_generator(train_data, train_conf['batch_size']), **train_conf['train_kwargs'])
        logger.info('Saving the model')
        model.save(os.path.join(output_folder, '%04d.h5' % epoch), include_optimizer=False)
        model.save(train_conf['model_path'], include_optimizer=False)


def play_matches(conf):
    command = 'python play_matches.py "%s" "%s" "%s" --n_matches %i' % (
        conf['output_folder'], conf['learning_agent'], conf['frozen_agent'], conf['n_matches']
    )
    os.system(command)


def load_train_data(folder):
    filepaths = sorted(glob.glob(os.path.join(folder, '*.json')))
    matches = [load_match_from_json(json_filepath, player=0) for json_filepath in tqdm(filepaths, desc='loading matches')]
    results = get_matches_results(filepaths)
    # invert_target_if_loss_match(matches, results) # this destroys the policy
    matches = leave_only_matches_with_wins(matches, results)
    if matches:
        return combine_data_for_training(matches, verbose=False), results
    else:
        return None, results


def get_matches_results(filepaths):
    results = []
    for filepath in tqdm(filepaths, desc='loading matches results'):
        with open(filepath, 'r') as f:
            match = json.load(f)
        last_step = match['steps'][-1]
        if last_step[0]['reward'] > last_step[1]['reward']:
            results.append(1)
        elif last_step[0]['reward'] < last_step[1]['reward']:
            results.append(-1)
        else:
            results.append(0)
    return np.array(results)


def invert_target_if_loss_match(matches, results):
    for match, result in zip (matches, results):
        if result < 0:
            for key in ['unit_output', 'city_output']:
                # invert everything except the last layer which is the mask
                match[key][:, :, :, :-1] = 1 - match[key][:, :, :, :-1]
                # I might change the mask to modify the loss


def leave_only_matches_with_wins(matches, results):
    return [match for match, result in zip (matches, results) if result > 0]


def log_matches_results(results, epoch, tensorboard_writer):
    metrics = {
        'win_rate': np.sum(results == 1)/len(results)*100,
        'tie_rate': np.sum(results == 0)/len(results)*100,
        'loss_rate': np.sum(results == -1)/len(results)*100,
        'n_matches': len(results),
    }
    for key, value in metrics.items():
        log_to_tensorboard(key, value, epoch, tensorboard_writer)
        logger.info('%s: %i' % (key, value))


def log_to_tensorboard(key, value, epoch, tensorboard_writer):
    with tensorboard_writer.as_default():
        tf.summary.scalar(key, value, step=epoch)


def parse_args(args):
    epilog = """
    """
    description = """
    Improves a policy by playing against frozen agents increasing the probabilities of the actions
    that lead to wins and decreasing those to lead to loses
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('config_path', help='Path to yaml file with training configuration')
    return parser.parse_args(args)


if __name__ == '__main__':
    main()

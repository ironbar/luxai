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
from train import create_model, create_callbacks
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
    update_train_conf_paths_with_output_folder(train_conf, output_folder)

    model = create_model(train_conf['model_params'])
    callbacks = create_callbacks(dict(), output_folder)
    for epoch in range(train_conf['max_epochs']):
        print()
        logger.info('Start epoch %i' % epoch)
        play_matches(train_conf['play_matches'])
        logger.info('Loading train data')
        train_data, results = load_train_data(
            train_conf['play_matches']['output_folder'], train_conf['data_collection_method'])
        log_matches_results(results, epoch, tensorboard_writer)
        logger.info('Fitting the model')
        model.fit(x=train_generator(train_data, train_conf['batch_size']), epochs=(epoch+1),
                  callbacks=callbacks, initial_epoch=epoch, **train_conf['train_kwargs'])
        del train_data
        logger.info('Saving the model')
        model.save(os.path.join(output_folder, '%04d.h5' % epoch), include_optimizer=False)
        model.save(train_conf['model_path'], include_optimizer=False)
        tf.keras.backend.clear_session()


def play_matches(conf):
    command = 'python play_matches.py "%s" "%s" "%s" --n_matches %i' % (
        conf['output_folder'], conf['learning_agent'], conf['frozen_agent'], conf['n_matches']
    )
    os.system(command)


def load_train_data(folder, method):
    filepaths = sorted(glob.glob(os.path.join(folder, '*.json')))
    results = get_matches_results(filepaths)
    if method == 'only_wins':
        matches = [load_match_from_json(json_filepath, player=0) for json_filepath in tqdm(filepaths, desc='loading matches')]
        matches = leave_only_matches_with_wins(matches, results)
        if matches:
            return combine_data_for_training(matches, verbose=False), results
        else:
            return None, results
    elif method == 'whoever_wins':
        matches = []
        iterator = tqdm(zip(filepaths, results), total=len(filepaths), desc='loading matches')
        for json_filepath, result in iterator:
            if result == 0:
                continue
            if result == 1:
                player = 0
            else:
                player = 1
            matches.append(load_match_from_json(json_filepath, player=player))
        return combine_data_for_training(matches, verbose=False), results
    elif method.startswith('wins_and_loses'):
        matches = [load_match_from_json(json_filepath, player=0) for json_filepath in tqdm(filepaths, desc='loading matches')]
        loses_weight = float(method.split('_')[-1])
        invert_target_if_loss_match(matches, results, loses_weight)
        return combine_data_for_training(matches, verbose=False), results
    elif method.startswith('all_data'):
        matches = [load_match_from_json(json_filepath, player=0) for json_filepath in tqdm(filepaths, desc='loading matches')]
        matches += [load_match_from_json(json_filepath, player=1) for json_filepath in tqdm(filepaths, desc='loading matches')]
        loses_weight = float(method.split('_')[-1])
        invert_target_if_loss_match(matches, np.concatenate([results, 1 - results]), loses_weight)
        return combine_data_for_training(matches, verbose=False), results


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


def invert_target_if_loss_match(matches, results, loses_weight):
    for match, result in zip (matches, results):
        if result < 0:
            for key in ['unit_output', 'city_output']:
                # invert everything except the last layer which is the mask
                match[key][..., :-1] = 1 - match[key][..., :-1]
                # Change the weight of the loses because loss value is much higher
                match[key][..., -1] *= loses_weight


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


def update_train_conf_paths_with_output_folder(train_conf, output_folder):
    key_pairs = [
        ('model_params', 'pretrained_weights'),
        ('play_matches', 'output_folder'),
        ('play_matches', 'learning_agent'),
    ]
    for key1, key2 in key_pairs:
        if not train_conf[key1][key2].startswith('/'):
            train_conf[key1][key2] = os.path.join(
                output_folder, train_conf[key1][key2]
            )
    for key in ['model_path']:
        if not train_conf[key].startswith('/'):
            train_conf[key] = os.path.join(output_folder, train_conf[key])


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

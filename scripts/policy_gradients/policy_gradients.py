import os
import sys
import argparse
import yaml
import logging
import glob
from tqdm import tqdm

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

    model = create_model(train_conf['model_params'])
    for epoch in range(train_conf['max_epochs']):
        print()
        logger.info('Start epoch %i' % epoch)
        play_matches(train_conf['play_matches'])
        logger.info('Loading train data')
        train_data = load_train_data(train_conf['play_matches']['output_folder'])
        logger.info('Fitting the model')
        model.fit(x=train_generator(train_data, train_conf['batch_size']), **train_conf['train_kwargs'])
        logger.info('Saving the model')
        model.save(os.path.join(output_folder, '%04d.h5' % epoch), include_optimizer=False)
        # fit model on matches
        # save model


def play_matches(conf):
    command = 'python play_matches.py "%s" "%s" "%s" --n_matches %i' % (
        conf['output_folder'], conf['learning_agent'], conf['frozen_agent'], conf['n_matches']
    )
    os.system(command)


def load_train_data(folder):
    filepaths = sorted(glob.glob(os.path.join(folder, '*.json')))
    matches = [load_match_from_json(json_filepath, player=0) for json_filepath in tqdm(filepaths, desc='loading matches')]
    # TODO: metrics about results, invert loses target
    return combine_data_for_training(matches, verbose=False)


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

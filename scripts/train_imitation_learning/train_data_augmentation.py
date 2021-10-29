import sys
import os
import yaml
import argparse
import numpy as np

import tensorflow as tf

from luxai.data import load_train_and_test_data
from luxai.data_augmentation import random_data_augmentation

from train import create_model, create_callbacks


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    train(args.config_path)


def train(config_path):
    with open(config_path) as f:
        train_conf = yaml.safe_load(f)
    output_folder = os.path.dirname(os.path.realpath(config_path))

    train_data, test_data = load_train_and_test_data(**train_conf['data'])
    model = create_model(train_conf['model_params'])
    callbacks = create_callbacks(train_conf['callbacks'], output_folder)
    train_conf['train_kwargs']['validation_batch_size'] = find_optimum_batch_size(
        len(test_data[0][0]), train_conf['train_kwargs']['validation_batch_size'] )
    model.fit(x=train_generator(train_data, train_conf['batch_size']), validation_data=test_data,
              callbacks=callbacks, **train_conf['train_kwargs'])


def train_generator(train_data, batch_size):
    indices = np.arange(len(train_data[0][0]))
    while 1:
        np.random.shuffle(indices)
        n_batches = len(indices)//batch_size
        for batch_idx in range(n_batches):
            batch_indices = indices[batch_idx*batch_size: (batch_idx+1)*batch_size]
            x = train_data[0][0][batch_indices], train_data[0][1][batch_indices]
            y = train_data[1][0][batch_indices], train_data[1][1][batch_indices]
            yield random_data_augmentation(x, y)


def find_optimum_batch_size(n_samples, ref_batch_size):
    candidate_batch_sizes = np.arange(ref_batch_size - 2, ref_batch_size + 3, dtype=int)
    return candidate_batch_sizes[np.argmax(n_samples % candidate_batch_sizes)]


def parse_args(args):
    epilog = """
    python train.py /mnt/hdd0/Kaggle/luxai/models/01_first_steps/train_conf.yml
    """
    description = """
    Train model for imitation learning with data augmentation
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('config_path', help='Path to the yaml file with train configuration')
    return parser.parse_args(args)


if __name__ == '__main__':
    main()

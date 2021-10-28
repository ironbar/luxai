import sys
import os
import yaml
import argparse
import numpy as np

import tensorflow as tf

from luxai.data import data_generator
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

    train_generator = data_generator(**train_conf['data']['train'])
    val_generator = data_generator(**train_conf['data']['val'])
    model = create_model(train_conf['model_params'])
    callbacks = create_callbacks(train_conf['callbacks'], output_folder)
    model.fit(x=train_generator, validation_data=val_generator,
              callbacks=callbacks, **train_conf['train_kwargs'])


def parse_args(args):
    epilog = """
    python train.py /mnt/hdd0/Kaggle/luxai/models/01_first_steps/train_conf.yml
    """
    description = """
    Train model for imitation learning with generators to reduce RAM requirements
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('config_path', help='Path to the yaml file with train configuration')
    return parser.parse_args(args)


if __name__ == '__main__':
    main()

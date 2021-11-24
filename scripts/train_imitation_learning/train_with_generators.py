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

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        train_enqueuer, val_enqueuer = get_enqueuers(train_conf['data'])
        model = create_model(train_conf['model_params'])
        callbacks = create_callbacks(train_conf['callbacks'], output_folder)
        model.fit(x=train_enqueuer.get(), validation_data=val_enqueuer.get(),
                  callbacks=callbacks, **train_conf['train_kwargs'])
        train_enqueuer.stop()
        val_enqueuer.stop()


def get_enqueuers(data_conf):
    enqueuers = []
    for generator_conf in [data_conf['train'], data_conf['val']]:
        generator = data_generator(**generator_conf)
        enqueuer = tf.keras.utils.GeneratorEnqueuer(generator)
        enqueuer.start(max_queue_size=data_conf['max_queue_size'])
        enqueuers.append(enqueuer)
    return enqueuers


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

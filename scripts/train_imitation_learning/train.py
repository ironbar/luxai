"""
Train model for imitation learning
"""
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
import json
import yaml
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
from kaggle_environments import make
import pandas as pd
from tqdm.notebook import tqdm

from luxai.cunet import cunet_luxai_model
from luxai.cunet import config as cunet_config
from luxai.data import load_train_and_test_data



def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    with open(args.config_path) as f:
        train_conf = yaml.safe_load(f)
    train(train_conf)


def train(train_conf):
    train_data, test_data = load_train_and_test_data(**train_conf['data'])
    model = create_model(train_conf['model_params'])
    # TODO: callbacks
    model.fit(x=train_data[0], y=train_data[1], validation_data=test_data, **train_conf['train_kwargs'])


def create_model(model_params: dict):
    # Unet parameters
    cunet_config.INPUT_SHAPE = model_params['board_shape']
    cunet_config.FILTERS_LAYER_1 = model_params['filters_layer_1']
    cunet_config.N_LAYERS = model_params['n_layers']
    cunet_config.ACT_LAST = model_params['act_last']
    # Condition parameters
    cunet_config.Z_DIM = model_params['z_dim']
    cunet_config.CONTROL_TYPE = model_params['control_type']
    cunet_config.FILM_TYPE = model_params['film_type']
    cunet_config.N_NEURONS = model_params['n_neurons']
    cunet_config.N_CONDITIONS = cunet_config.N_LAYERS # 6 this should be the same as the number of layers
    # Other
    cunet_config.LR = 1e-3 # 1e-3

    model = cunet_luxai_model(cunet_config)
    return model


def parse_args(args):
    epilog = """
    python train.py /mnt/hdd0/Kaggle/luxai/models/01_first_steps/train_conf.yml
    """
    description = """
    Train model for imitation learning
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('config_path', help='Path to the yaml file with train configuration')
    return parser.parse_args(args)


if __name__ == '__main__':
    main()

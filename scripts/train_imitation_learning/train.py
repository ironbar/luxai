"""
Train model for imitation learning
"""
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
import yaml
import argparse
import tensorflow as tf
from scipy.interpolate import interp1d

from luxai.cunet import cunet_luxai_model
from luxai.cunet import config as cunet_config
from luxai.data import load_train_and_test_data
from luxai.callbacks import (
    LogLearningRate, LogEpochTime, LogRAM, LogCPU, LogGPU, GarbageCollector
)


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
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    model = create_model(train_conf['model_params'])
    callbacks = create_callbacks(train_conf['callbacks'], output_folder)
    model.fit(x=train_data[0], y=train_data[1], validation_data=test_data, callbacks=callbacks,
            **train_conf['train_kwargs'])


def create_callbacks(callbacks_conf, output_folder):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(output_folder, 'logs'), profile_batch=0)
    tensorboard_callback._supports_tf_logs = False
    callbacks = [
        LogEpochTime(),
        LogLearningRate(),
        LogRAM(),
        LogCPU(),
        LogGPU(),
        tensorboard_callback,
        GarbageCollector(),
    ]
    if 'EarlyStopping' in callbacks_conf:
        callbacks.append(tf.keras.callbacks.EarlyStopping(**callbacks_conf['EarlyStopping']))
    for name, callback_kwargs in callbacks_conf.items():
        if 'ModelCheckpoint' in name:
            callback_kwargs['filepath'] = callback_kwargs['filepath'] % output_folder
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(**callback_kwargs))
    if 'ReduceLROnPlateau' in callbacks_conf:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(**callbacks_conf['ReduceLROnPlateau']))
    if 'LearningRateScheduler' in callbacks_conf:
        schedule = interp1d(**callbacks_conf['LearningRateScheduler'])
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(schedule=lambda x: float(schedule(x))))
    return callbacks


def create_model(model_params: dict):
    # Unet parameters
    cunet_config.INPUT_SHAPE = model_params['board_shape']
    if 'layer_filters' in model_params:
        cunet_config.layer_filters = model_params['layer_filters']
        cunet_config.final_layer_filters = model_params['final_layer_filters']
    else:
        cunet_config.FILTERS_LAYER_1 = model_params['filters_layer_1']
        cunet_config.N_LAYERS = model_params['n_layers']
    # Condition parameters
    cunet_config.Z_DIM = model_params['z_dim']
    cunet_config.CONTROL_TYPE = model_params['control_type']
    cunet_config.FILM_TYPE = model_params['film_type']
    cunet_config.N_NEURONS = model_params['n_neurons']
    # Other
    cunet_config.LR = model_params['lr']
    # cunet_config.loss_name = model_params['loss']
    # cunet_config.loss_kwargs = model_params['loss_kwargs']
    cunet_config.loss_weights = model_params.get('loss_weights', None)
    cunet_config.freeze_bn_layers = model_params.get('freeze_bn_layers', False)
    cunet_config.dropout = model_params['dropout']
    cunet_config.regularizer_kwargs = model_params.get('regularizer_kwargs', None)

    model = cunet_luxai_model(cunet_config)
    model.summary()
    if 'pretrained_weights' in model_params:
        print('Loading model weights from: %s' % model_params['pretrained_weights'])
        model.load_weights(model_params['pretrained_weights'], skip_mismatch=True, by_name=True)
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

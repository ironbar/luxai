"""
Customized version of cunet adapted from https://github.com/gabolsgabs/cunet
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization
)

import sys
sys.path.append('/mnt/hdd0/MEGA/AI/22 Kaggle/luxai/forum/cunet')

from cunet.train.models.cunet_model import (
    u_net_conv_block, dense_control, cnn_control, u_net_deconv_block,
    slice_tensor, slice_tensor_range, cunet_model, config
)

def cunet_luxai_model(config):
    board_input = Input(shape=config.INPUT_SHAPE, name='board_input')
    n_layers = config.N_LAYERS
    x = board_input
    encoder_layers = []
    initializer = tf.random_normal_initializer(stddev=0.02)

    if config.CONTROL_TYPE == 'dense':
        input_conditions, gammas, betas = dense_control(
            n_conditions=config.N_CONDITIONS, n_neurons=config.N_NEURONS)
    if config.CONTROL_TYPE == 'cnn':
        input_conditions, gammas, betas = cnn_control(
            n_conditions=config.N_CONDITIONS, n_filters=config.N_FILTERS)
    # Encoder
    complex_index = 0
    for i in range(n_layers):
        n_filters = config.FILTERS_LAYER_1 * (2 ** i)
        if config.FILM_TYPE == 'simple':
            gamma, beta = slice_tensor(i)(gammas), slice_tensor(i)(betas)
        if config.FILM_TYPE == 'complex':
            init, end = complex_index, complex_index+n_filters
            gamma = slice_tensor_range(init, end)(gammas)
            beta = slice_tensor_range(init, end)(betas)
            complex_index += n_filters
        if i == 0: # do not reduce dimensionality on the first encoding layer
            strides = (1, 1)
        else:
            strides = (2, 2)
        x = u_net_conv_block(
            x, n_filters, initializer, gamma, beta,
            kernel_size=(3, 3), strides=strides,
            activation=config.ACTIVATION_ENCODER, film_type=config.FILM_TYPE
        )
        encoder_layers.append(x)
    # Decoder
    for i in range(n_layers):
        is_final_block = i == n_layers - 1  # the las layer is different
        # not dropout in the first block and the last two encoder blocks
        dropout = not (i == 0 or i == n_layers - 1 or i == n_layers - 2)
        # for getting the number of filters
        encoder_layer = encoder_layers[n_layers - i - 1]
        skip = i > 0    # not skip in the first encoder block

        n_filters = encoder_layer.get_shape().as_list()[-1] // 2
        activation = config.ACTIVATION_DECODER
        if is_final_block: # do not reduce dimensionality on the first encoding layer
            strides = (1, 1)
        else:
            strides = (2, 2)
        x = u_net_deconv_block(
            x, encoder_layer, n_filters, initializer, activation, dropout, skip,
            kernel_size=(3, 3), strides=strides,
        )
    outputs = [
        Conv2D(filters=10, kernel_size=1, activation=config.ACT_LAST, name='unit_action')(x),
        Conv2D(filters=3, kernel_size=1, activation=config.ACT_LAST, name='city_action')(x)
    ]
    model = Model(inputs=[board_input, input_conditions], outputs=outputs)
    model.compile(
        optimizer=Adam(lr=config.LR, beta_1=0.5), loss=config.LOSS)
        # experimental_run_tf_function=False)
    return model

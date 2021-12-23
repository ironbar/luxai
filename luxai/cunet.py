"""
Customized version of cunet adapted from https://github.com/gabolsgabs/cunet
"""
import os
from functools import partial
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization
)
import tensorflow.keras.backend as K

import sys
KERAS_CUNET_PATH = os.environ.get('KERAS_CUNET_PATH', '/mnt/hdd0/MEGA/AI/22 Kaggle/luxai/forum/cunet')
sys.path.append(KERAS_CUNET_PATH)

from cunet.train.models.cunet_model import (
    u_net_conv_block, dense_control, cnn_control, u_net_deconv_block,
    slice_tensor, slice_tensor_range, cunet_model, config
)

from luxai.metrics import (
    masked_binary_crossentropy, masked_categorical_crossentropy,
    masked_error, masked_categorical_error
)

def cunet_luxai_model(config):
    board_input = Input(shape=config.INPUT_SHAPE, name='board_input')
    x = board_input
    encoder_layers = []
    initializer = tf.random_normal_initializer(stddev=0.02)

    if hasattr(config, 'layer_filters'):
        layer_filters = config.layer_filters
        n_layers = len(layer_filters)
        final_layer_filters = config.final_layer_filters # with this option this is configurable
    else:
        n_layers = config.N_LAYERS
        layer_filters = [config.FILTERS_LAYER_1 * (2 ** i) for i in range(n_layers)]
        final_layer_filters = layer_filters[0]//2 # this was hardcoded

    if config.FILM_TYPE == 'simple':
        config.N_CONDITIONS = len(layer_filters)
    else:
        config.N_CONDITIONS = sum(layer_filters)

    if config.CONTROL_TYPE == 'dense':
        input_conditions, gammas, betas = dense_control(
            n_conditions=config.N_CONDITIONS, n_neurons=config.N_NEURONS)
    if config.CONTROL_TYPE == 'cnn':
        input_conditions, gammas, betas = cnn_control(
            n_conditions=config.N_CONDITIONS, n_filters=config.N_FILTERS)
    # Encoder
    complex_index = 0
    for i, n_filters in enumerate(layer_filters):
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
    layer_filters.insert(0, final_layer_filters)
    # Decoder
    for i in range(n_layers):
        is_final_block = i == n_layers - 1  # the las layer is different
        # not dropout in the first block and the last two encoder blocks
        dropout = config.dropout[i]
        # for getting the number of filters
        encoder_layer = encoder_layers[n_layers - i - 1]
        n_filters = layer_filters[n_layers - i - 1]
        skip = i > 0    # not skip in the first encoder block
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
        Conv2D(filters=1, kernel_size=1, activation='sigmoid', name='unit_action')(x),
        Conv2D(filters=10, kernel_size=1, activation='softmax', name='unit_policy')(x),
        Conv2D(filters=1, kernel_size=1, activation='sigmoid', name='city_action')(x),
        Conv2D(filters=3, kernel_size=1, activation='softmax', name='city_policy')(x)
    ]
    model = Model(inputs=[board_input, input_conditions], outputs=outputs)
    model.save = partial(model.save, include_optimizer=False) # this allows to use ModelCheckpoint callback, otherwise fails to serialize the loss function

    if config.freeze_bn_layers:
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                print('Freezing layer: %s' % str(layer))
                layer.trainable = False

    # adding regularization
    if config.regularizer_kwargs is not None:
        print('Adding regularization: %s' % str(config.regularizer_kwargs))
        regularizer = tf.keras.regularizers.L1L2(**config.regularizer_kwargs)
        for layer in model.layers:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    layer.add_loss(lambda layer=layer: regularizer(layer.kernel))
                    print(layer.name)
                    # setattr(layer, attr, regularizer)

    model.compile(
        optimizer=Adam(lr=config.LR, beta_1=0.5),
        # loss=get_loss_function(config.loss_name, config.loss_kwargs),
        loss = {
            'unit_action': masked_binary_crossentropy,
            'city_action': masked_binary_crossentropy,
            'unit_policy': masked_categorical_crossentropy,
            'city_policy': masked_categorical_crossentropy,
        },
        metrics = {
            'unit_action': masked_error,
            'city_action': masked_error,
            'unit_policy': masked_categorical_error,
            'city_policy': masked_categorical_error,
        },
        # metrics=[masked_error, true_positive_error, true_negative_error],
        loss_weights=config.loss_weights,
    )
    return model

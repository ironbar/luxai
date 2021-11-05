"""
Customized version of cunet adapted from https://github.com/gabolsgabs/cunet
"""
from functools import partial
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization
)
import tensorflow.keras.backend as K

import sys
sys.path.append('/mnt/hdd0/MEGA/AI/22 Kaggle/luxai/forum/cunet')

from cunet.train.models.cunet_model import (
    u_net_conv_block, dense_control, cnn_control, u_net_deconv_block,
    slice_tensor, slice_tensor_range, cunet_model, config
)

def cunet_luxai_model(config):
    board_input = Input(shape=config.INPUT_SHAPE, name='board_input')
    x = board_input
    encoder_layers = []
    initializer = tf.random_normal_initializer(stddev=0.02)

    if hasattr(config, 'layer_filters'):
        layer_filters = config.layer_filters
        n_layers = len(layer_filters)
    else:
        n_layers = config.N_LAYERS
        layer_filters = [config.FILTERS_LAYER_1 * (2 ** i) for i in range(n_layers)]

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
    model.save = partial(model.save, include_optimizer=False) # this allows to use ModelCheckpoint callback, otherwise fails to serialize the loss function
    model.compile(
        optimizer=Adam(lr=config.LR, beta_1=0.5),
        loss=get_loss_function(config.loss_name, config.loss_kwargs),
        metrics=[masked_error, true_positive_error, true_negative_error],
        loss_weights=config.loss_weights,
    )
    return model


def get_loss_function(loss_name, kwargs):
    name_to_loss = {
        'masked_binary_crossentropy': masked_binary_crossentropy,
        'masked_focal_loss': masked_focal_loss,
    }
    return partial(name_to_loss[loss_name], **kwargs)


def masked_binary_crossentropy(y_true, y_pred, true_weight=1):
    mask, labels = _split_y_true_on_labels_and_mask(y_true)
    loss = custom_binary_crossentropy(labels, y_pred, true_weight=true_weight)
    return apply_mask_to_loss(loss, mask)


def apply_mask_to_loss(loss, mask):
    return K.sum(loss*mask)/(K.sum(mask)*K.cast_to_floatx(K.shape(loss)[-1]))


def custom_binary_crossentropy(y_true, y_pred, true_weight=1):
    """https://github.com/keras-team/keras/blob/3a33d53ea4aca312c5ad650b4883d9bac608a32e/keras/backend.py#L5014"""
    bce = true_weight*y_true*K.log(y_pred + K.epsilon()) + (1 - y_true)*K.log(1 - y_pred + K.epsilon())
    return -bce


def masked_error(y_true, y_pred):
    mask, labels = _split_y_true_on_labels_and_mask(y_true)
    accuracy = K.cast_to_floatx(labels == K.round(y_pred))
    error = 1 - accuracy
    return apply_mask_to_loss(error, mask)


def true_positive_error(y_true, y_pred):
    return true_generic_error(y_true, y_pred, label=1)


def true_negative_error(y_true, y_pred):
    return true_generic_error(y_true, y_pred, label=0)


def true_generic_error(y_true, y_pred, label):
    mask, labels = _split_y_true_on_labels_and_mask(y_true)
    mask = mask * K.cast_to_floatx(labels == label)
    accuracy = K.cast_to_floatx(labels == K.round(y_pred))
    error = 1 - accuracy
    return apply_mask_to_loss(error, mask)


def _split_y_true_on_labels_and_mask(y_true):
    mask = y_true[:, :, :, -1:]
    y_true = y_true[:, :, :, :-1]
    return mask, y_true


def masked_focal_loss(y_true, y_pred, zeta=1, true_weight=1):
    mask, labels = _split_y_true_on_labels_and_mask(y_true)
    return apply_mask_to_loss(focal_loss(labels, y_pred, zeta, true_weight=true_weight), mask)


def focal_loss(y_true, y_pred, zeta, true_weight=1):
    """https://github.com/umbertogriffo/focal-loss-keras"""
    loss = true_weight*y_true*K.log(y_pred + K.epsilon())*(1 - y_pred + K.epsilon())**zeta
    loss += (1 - y_true)*K.log(1 - y_pred + K.epsilon())*(y_pred + K.epsilon())**zeta
    return -loss

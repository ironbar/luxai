"""
Custom metrics and losses
"""
from functools import partial
import tensorflow.keras.backend as K


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


def masked_categorical_crossentropy(y_true, y_pred):
    mask = _get_mask_for_categorical(y_true)
    loss = -y_true*K.log(y_pred + K.epsilon())
    return apply_mask_to_loss(loss, mask)


def _get_mask_for_categorical(y_true):
    return K.expand_dims(K.max(y_true, axis=-1), axis=-1)


def apply_mask_to_loss(loss, mask):
    # return K.sum(loss*mask)/(K.sum(mask)*K.cast_to_floatx(K.shape(loss)[-1]))
    return K.sum(loss*mask)/K.sum(mask)


def custom_binary_crossentropy(y_true, y_pred, true_weight=1):
    """https://github.com/keras-team/keras/blob/3a33d53ea4aca312c5ad650b4883d9bac608a32e/keras/backend.py#L5014"""
    bce = true_weight*y_true*K.log(y_pred + K.epsilon()) + (1 - y_true)*K.log(1 - y_pred + K.epsilon())
    return -bce


def masked_error(y_true, y_pred):
    mask, labels = _split_y_true_on_labels_and_mask(y_true)
    accuracy = K.cast_to_floatx(labels == K.round(y_pred))
    error = 1 - accuracy
    return apply_mask_to_loss(error, mask)


def masked_categorical_error(y_true, y_pred):
    """https://github.com/keras-team/keras/blob/2c48a3b38b6b6139be2da501982fd2f61d7d48fe/keras/metrics.py#L3544"""
    mask = _get_mask_for_categorical(y_true)
    error = K.expand_dims(categorical_error(y_true, y_pred), axis=-1)
    return apply_mask_to_loss(error, mask)


def categorical_error(y_true, y_pred):
    accuracy = K.cast_to_floatx(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))
    error = 1 - accuracy
    return error


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
    mask = y_true[..., -1:]
    y_true = y_true[..., :-1]
    return mask, y_true


def masked_focal_loss(y_true, y_pred, zeta=1, true_weight=1):
    mask, labels = _split_y_true_on_labels_and_mask(y_true)
    return apply_mask_to_loss(focal_loss(labels, y_pred, zeta, true_weight=true_weight), mask)


def focal_loss(y_true, y_pred, zeta, true_weight=1):
    """https://github.com/umbertogriffo/focal-loss-keras"""
    loss = true_weight*y_true*K.log(y_pred + K.epsilon())*(1 - y_pred + K.epsilon())**zeta
    loss += (1 - y_true)*K.log(1 - y_pred + K.epsilon())*(y_pred + K.epsilon())**zeta
    return -loss

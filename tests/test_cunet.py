import pytest
import numpy as np

from luxai.cunet import masked_binary_crossentropy, masked_error, cunet_luxai_model, config


@pytest.mark.parametrize('y_pred, y_true, mask, loss', [
    ([[[[1, 1]]]], [[[[1, 1]]]], [[[[1]]]], 0), # shape 1, 1, 1, 2
    ([[[[0.5, 1]]]], [[[[1, 1]]]], [[[[1]]]], -np.log(0.5)/2), # shape 1, 1, 1, 2
    ([[[[0.5, 0.5]]]], [[[[1, 1]]]], [[[[1]]]], -np.log(0.5)), # shape 1, 1, 1, 2
    ([[[[1, 1], [1, 1]]]], [[[[1, 1], [1, 1]]]], [[[[1], [1]]]], 0), # shape 1, 1, 2, 2
    ([[[[1, 1], [0.5, 0.5]]]], [[[[1, 1], [1, 1]]]], [[[[1], [1]]]], -np.log(0.5)/2), # shape 1, 1, 2, 2
    # play with the mask
    ([[[[1], [0.5]]]], [[[[1], [1]]]], [[[[1], [1]]]], -np.log(0.5)/2), # shape 1, 1, 2, 1
    ([[[[1], [0.5]]]], [[[[1], [1]]]], [[[[0], [1]]]], -np.log(0.5)), # shape 1, 1, 2, 1
    ([[[[1], [0.5]]]], [[[[1], [1]]]], [[[[1], [0]]]], 0), # shape 1, 1, 2, 1
    # verify that changing the shape does not modify the result
    ([[[[1, 1, 1]]]], [[[[1, 1, 1]]]], [[[[1]]]], 0), # shape 1, 1, 1, 3
    ([[[[1, 1, 1, 1]]]], [[[[1, 1, 1, 1]]]], [[[[1]]]], 0), # shape 1, 1, 1, 4
    ([[[[1, 1, 1, 1, 1]]]], [[[[1, 1, 1, 1, 1]]]], [[[[1]]]], 0), # shape 1, 1, 1, 5
    ([[[[0.5, 0.5, 0.5]]]], [[[[1, 1, 1]]]], [[[[1]]]], -np.log(0.5)), # shape 1, 1, 1, 3
    ([[[[0.5, 0.5, 0.5, 0.5]]]], [[[[1, 1, 1, 1]]]], [[[[1]]]], -np.log(0.5)), # shape 1, 1, 1, 4
    ([[[[0.5, 0.5, 0.5, 0.5, 0.5]]]], [[[[1, 1, 1, 1, 1]]]], [[[[1]]]], -np.log(0.5)), # shape 1, 1, 1, 5
])
def test_masked_binary_crossentropy(y_pred, y_true, mask, loss):
    y_true = np.concatenate([np.array(y_true, dtype=np.float32), np.array(mask, dtype=np.float32)], axis=-1)
    y_pred = np.array(y_pred, dtype=np.float32)
    assert pytest.approx(loss, abs=1e-6) == masked_binary_crossentropy(y_true, y_pred)


@pytest.mark.parametrize('y_pred, y_true, mask, loss', [
    ([[[[1, 1]]]], [[[[1, 1]]]], [[[[1]]]], 0), # shape 1, 1, 1, 2
    ([[[[0.9, 0.7]]]], [[[[1, 1]]]], [[[[1]]]], 0), # shape 1, 1, 1, 2
    ([[[[0, 1]]]], [[[[1, 1]]]], [[[[1]]]], 0.5), # shape 1, 1, 1, 2
    ([[[[0, 0]]]], [[[[1, 1]]]], [[[[1]]]], 1.0), # shape 1, 1, 1, 2
    ([[[[1, 1], [1, 1]]]], [[[[1, 1], [1, 1]]]], [[[[1], [1]]]], 0), # shape 1, 1, 2, 2
    ([[[[1, 1], [0, 0]]]], [[[[1, 1], [1, 1]]]], [[[[1], [1]]]], 0.5), # shape 1, 1, 2, 2
    ([[[[1, 1], [0.1, 0.2]]]], [[[[1, 1], [1, 1]]]], [[[[1], [1]]]], 0.5), # shape 1, 1, 2, 2
    # # play with the mask
    ([[[[1], [0]]]], [[[[1], [1]]]], [[[[1], [1]]]], 0.5), # shape 1, 1, 2, 1
    ([[[[1], [0]]]], [[[[1], [1]]]], [[[[0], [1]]]], 1), # shape 1, 1, 2, 1
    ([[[[1], [0]]]], [[[[1], [1]]]], [[[[1], [0]]]], 0), # shape 1, 1, 2, 1
    # # verify that changing the shape does not modify the result
    ([[[[1, 1, 1]]]], [[[[1, 1, 1]]]], [[[[1]]]], 0), # shape 1, 1, 1, 3
    ([[[[1, 1, 1, 1]]]], [[[[1, 1, 1, 1]]]], [[[[1]]]], 0), # shape 1, 1, 1, 4
    ([[[[1, 1, 1, 1, 1]]]], [[[[1, 1, 1, 1, 1]]]], [[[[1]]]], 0), # shape 1, 1, 1, 5
    ([[[[0, 0, 0]]]], [[[[1, 1, 1]]]], [[[[1]]]], 1), # shape 1, 1, 1, 3
    ([[[[0, 0, 0, 0]]]], [[[[1, 1, 1, 1]]]], [[[[1]]]], 1), # shape 1, 1, 1, 4
    ([[[[0, 0, 0, 0, 0]]]], [[[[1, 1, 1, 1, 1]]]], [[[[1]]]], 1), # shape 1, 1, 1, 5
])
def test_masked_error(y_pred, y_true, mask, loss):
    y_true = np.concatenate([np.array(y_true, dtype=np.float32), np.array(mask, dtype=np.float32)], axis=-1)
    y_pred = np.array(y_pred, dtype=np.float32)
    assert pytest.approx(loss) == masked_error(y_true, y_pred)


def test_cunet_model_changes_when_modifying_both_inputs():
    # Unet parameters
    config.INPUT_SHAPE = [32, 32, 22] #[512, 128, 1]
    config.FILTERS_LAYER_1 = 32 # 16
    config.N_LAYERS = 3 # 6
    config.ACT_LAST = 'sigmoid' # sigmoid
    # Condition parameters
    config.Z_DIM = 12 # 4
    config.CONTROL_TYPE = 'dense' # dense
    config.FILM_TYPE = 'simple' # simple
    config.N_NEURONS = [16] # [16, 64, 256]
    config.N_CONDITIONS = config.N_LAYERS # 6 this should be the same as the number of layers
    config.loss_name = 'masked_binary_crossentropy'
    config.loss_kwargs = dict()


    model = cunet_luxai_model(config)
    np.random.seed(7)
    inputs = [np.random.normal(size=([1] + config.INPUT_SHAPE)), np.random.normal(size=([1, 1, config.Z_DIM]))]
    pred = model.predict(inputs)[0]

    inputs_2 = [inputs[0], np.random.normal(size=([1, 1, config.Z_DIM]))]
    pred_2 = model.predict(inputs_2)[0]
    assert pytest.approx(pred) != pred_2


    inputs_3 = [np.random.normal(size=([1] + config.INPUT_SHAPE)), inputs[1]]
    pred_3 = model.predict(inputs_3)[0]
    assert pytest.approx(pred) != pred_3

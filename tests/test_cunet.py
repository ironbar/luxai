import pytest
import numpy as np

from luxai.cunet import cunet_luxai_model, config


@pytest.mark.parametrize('film_type', ['simple', 'complex'])
def test_cunet_model_changes_when_modifying_both_inputs(film_type):
    # Unet parameters
    config.INPUT_SHAPE = [32, 32, 22] #[512, 128, 1]
    config.FILTERS_LAYER_1 = 4 # 16
    config.N_LAYERS = 3 # 6
    config.ACT_LAST = 'sigmoid' # sigmoid
    # Condition parameters
    config.Z_DIM = 12 # 4
    config.CONTROL_TYPE = 'dense' # dense
    config.FILM_TYPE = film_type # simple
    config.N_NEURONS = [16] # [16, 64, 256]
    config.loss_name = 'masked_binary_crossentropy'
    config.loss_kwargs = dict()
    config.loss_weights = None
    config.freeze_bn_layers = False


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


@pytest.mark.parametrize('layer_filters', [
    [32, 32, 64],
    [4, 4, 4, 4],
])
@pytest.mark.parametrize('film_type', ['simple', 'complex'])
def test_cunet_creation_with_custom_layer_filters(film_type, layer_filters):
    # Unet parameters
    config.INPUT_SHAPE = [32, 32, 22] #[512, 128, 1]
    config.layer_filters = layer_filters
    config.final_layer_filters = layer_filters[0]
    config.ACT_LAST = 'sigmoid' # sigmoid
    # Condition parameters
    config.Z_DIM = 12 # 4
    config.CONTROL_TYPE = 'dense' # dense
    config.FILM_TYPE = film_type # simple
    config.N_NEURONS = [16] # [16, 64, 256]
    config.loss_name = 'masked_binary_crossentropy'
    config.loss_kwargs = dict()
    config.loss_weights = None
    config.freeze_bn_layers = False

    model = cunet_luxai_model(config)
"""
Agent code for cunet model
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import numpy as np
import cunet.train.models.FiLM_utils

try:
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(SCRIPT_DIR, 'model.h5')
except NameError:
    # this happens when using python for playing matches
    # original model path: __replace_original_model_path__
    model_path = '__replace_model_path__'
model = tf.keras.models.load_model(model_path, compile=False)


def agent(observation, configuration):
    ret = make_input(observation)
    board, features = ret[:2]
    preds = model.predict([
        expand_board_size_adding_zeros(np.expand_dims(board, axis=0)),
        np.expand_dims(features, axis=0)])
    preds = [crop_board_to_original_size(pred, observation) for pred in preds]
    active_units_to_position, active_cities_to_position, units_to_position = ret[2:]
    actions = create_actions_for_units_from_model_predictions(
        preds[0][0], active_units_to_position, units_to_position, observation)
    actions += create_actions_for_cities_from_model_predictions(preds[1][0], active_cities_to_position)
    return actions

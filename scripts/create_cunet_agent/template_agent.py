"""
Agent code for cunet model
"""
import os
import glob
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import numpy as np
import cunet.train.models.FiLM_utils

# original model paths: __replace_original_model_path__
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
model_paths = sorted(glob.glob(os.path.join(SCRIPT_DIR, '*.h5')))
models = [tf.keras.models.load_model(model_path, compile=False) for model_path in model_paths]


def agent(observation, configuration):
    ret = make_input(observation)
    board, features = ret[:2]
    model_input = [expand_board_size_adding_zeros(np.expand_dims(board, axis=0)),
                   np.expand_dims(features, axis=0)]
    preds = [model.predict(model_input) for model in models]
    preds = [np.sum([pred[idx] for pred in preds], axis=0) for idx in range(2)]
    preds = [crop_board_to_original_size(pred, observation) for pred in preds]
    active_unit_to_position, active_city_to_position, unit_to_position, city_to_position = ret[2:]
    actions = create_actions_for_units_from_model_predictions(
        preds[0][0], active_unit_to_position, unit_to_position, observation, set(city_to_position.keys()))
    actions += create_actions_for_cities_from_model_predictions(preds[1][0], active_city_to_position)
    return actions

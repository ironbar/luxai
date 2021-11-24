import pytest
import numpy as np

from luxai.metrics import (
    masked_binary_crossentropy, masked_error, focal_loss,
    categorical_error, masked_categorical_error)


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


@pytest.mark.parametrize('y_pred, y_true, zeta, loss', [
    (1, 1, 0, 0),
    (0.75, 1, 0, -np.log(0.75)),
    (0.75, 1, 1, -np.log(0.75)*0.25),
    (0.75, 1, 2, -np.log(0.75)*0.25**2),
    (0.25, 0, 2, -np.log(0.75)*0.25**2),
    (0.25, 0, 1, -np.log(0.75)*0.25**1),
    (0.25, 0, 0, -np.log(0.75)),
])
def test_focal_loss(y_pred, y_true, zeta, loss):
    assert pytest.approx(loss, abs=1e-6) == focal_loss(y_true, y_pred, zeta)

@pytest.mark.parametrize('y_pred, y_true, error,', [
    ([[1, 0, 0]], [[1, 0, 0]], [0]),
    ([[1, 0, 0]], [[0, 1, 0]], [1]),
    ([[1, 0, 0], [1, 0, 0]], [[1, 0, 0], [1, 0, 0]], [0, 0]),
    ([[1, 0, 0], [1, 0, 0]], [[1, 0, 0], [0, 1, 0]], [0, 1]),
])
def test_categorical_error(y_true, y_pred, error):
    y_true, y_pred, error = np.array(y_true), np.array(y_pred), np.array(error)
    assert pytest.approx(error) == categorical_error(y_true, y_pred)


@pytest.mark.parametrize('y_true, y_pred, error,', [
    ([[1, 0, 0, 1]], [[1, 0, 0]], 0),
    ([[1, 0, 0, 1]], [[0, 1, 0]], 1),
    ([[1, 0, 0, 1], [1, 0, 0, 1]], [[1, 0, 0], [1, 0, 0]], 0),
    ([[1, 0, 0, 1], [1, 0, 0, 1]], [[1, 0, 0], [0, 1, 0]], 0.5),
    ([[1, 0, 0, 1], [1, 0, 0, 0]], [[1, 0, 0], [0, 1, 0]], 0),
    ([[1, 0, 0, 0], [1, 0, 0, 1]], [[1, 0, 0], [0, 1, 0]], 1),
])
def test_masked_categorical_error(y_true, y_pred, error):
    y_true, y_pred, error = np.array(y_true, dtype=np.float32), np.array(y_pred, dtype=np.float32), np.array(error, dtype=np.float32)
    print(y_true.shape, y_pred.shape)
    assert pytest.approx(error) == masked_categorical_error(y_true, y_pred)

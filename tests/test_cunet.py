import pytest
import numpy as np

from luxai.cunet import masked_binary_crossentropy


@pytest.mark.parametrize('y_pred, y_true, mask, loss', [
    ([[[[1, 1]]]], [[[[1, 1]]]], [[[[1]]]], 0), # shape 1, 1, 1, 2
    ([[[[0.5, 1]]]], [[[[1, 1]]]], [[[[1]]]], -np.log(0.5)/2), # shape 1, 1, 1, 2
    ([[[[0.5, 0.5]]]], [[[[1, 1]]]], [[[[1]]]], -np.log(0.5)), # shape 1, 1, 1, 2
    ([[[[1, 1], [1, 1]]]], [[[[1, 1], [1, 1]]]], [[[[1], [1]]]], 0), # shape 1, 1, 2, 2
    ([[[[1, 1], [0.5, 0.5]]]], [[[[1, 1], [1, 1]]]], [[[[1], [1]]]], -np.log(0.5)/2), # shape 1, 1, 2, 2
    ([[[[1], [0.5]]]], [[[[1], [1]]]], [[[[1], [1]]]], -np.log(0.5)/2), # shape 1, 1, 2, 1
    ([[[[1], [0.5]]]], [[[[1], [1]]]], [[[[0], [1]]]], -np.log(0.5)), # shape 1, 1, 2, 1
    ([[[[1], [0.5]]]], [[[[1], [1]]]], [[[[1], [0]]]], 0), # shape 1, 1, 2, 1
])
def test_masked_binary_crossentropy(y_pred, y_true, mask, loss):
    y_true = np.concatenate([np.array(y_true, dtype=np.float32), np.array(mask, dtype=np.float32)], axis=-1)
    y_pred = np.array(y_pred, dtype=np.float32)
    assert pytest.approx(loss) == masked_binary_crossentropy(y_true, y_pred)

import pytest
import numpy as np

from luxai.data_augmentation import horizontal_flip

@pytest.fixture
def batch():
    batch_size = 10
    x = (np.random.randint(0, 2, size=(batch_size, 32, 32, 24)),
         np.random.randint(0, 2, size=(batch_size, 1, 13)))
    y = (np.random.randint(0, 2, size=(batch_size, 32, 32, 11)),
         np.random.randint(0, 2, size=(batch_size, 32, 32, 4)))
    return x, y


def test_horizontal_flip_is_invertible(batch):
    _assert_batchs_are_equal(batch, horizontal_flip(*horizontal_flip(*batch)))


def _assert_batchs_are_equal(batch1, batch2):
    for i in range(2):
        for j in range(2):
            assert (batch1[i][j] == batch2[i][j]).all()


def test_horizontal_flip_modifies_everything_except_context(batch):
    _assert_batchs_are_different_except_from_context(batch, horizontal_flip(*batch))


def _assert_batchs_are_different_except_from_context(batch1, batch2):
    for i in range(2):
        for j in range(2):
            if i == 0 and j == 1:
                assert (batch1[i][j] == batch2[i][j]).all()
            else:
                assert not (batch1[i][j] == batch2[i][j]).all(), (i, j)

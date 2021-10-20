import pytest
import numpy as np

from luxai.input_features import expand_board_size_adding_zeros

@pytest.mark.parametrize('board', [
    np.zeros((1, 12, 12, 3)),
    np.zeros((1, 16, 16, 3)),
    np.zeros((1, 32, 32, 3))
])
def test_expand_board_size_adding_zeros(board, size=32):
    expanded_board = expand_board_size_adding_zeros(board)
    assert expanded_board.shape[0] == board.shape[0]
    assert expanded_board.shape[1] == size
    assert expanded_board.shape[2] == size
    assert expanded_board.shape[3] == board.shape[3]

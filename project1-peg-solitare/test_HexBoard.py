import pytest
from HexBoard import createHexGrid
import Node from HexBoard
from mock import MagickMock


def testHeaxBoard():
    board = createHexGrid(3)
    node = MagickMock(Node)
    for l in board:
        print(l)

    expected = [
        [node],
        [node, node],
        [node, node, node]
    ]
    assert expected == createHexGrid(3)


"""  class Test(TestCase):

    def test_create_hex_grid(self):
        node = mock.Mock()
        grid = (createHexGrid(3))
        for list in grid:
            print(grid) """

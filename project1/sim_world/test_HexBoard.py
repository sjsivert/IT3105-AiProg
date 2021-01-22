#import pytest
import unittest
from typing import Any
# ofrom mocker import mocker
#from unittest.mock import Mock
from unittest.mock import MagicMixin, MagicMock


class HexBoardTest(unittest.TestCase):
    def testHeaxBoardWithSize3(self):
        board = createHexGrid(3)
        node = MagicMock()
        print(board)
        expected = [
            [Node()],
            [Node(), Node()],
            [Node(), Node(), Node()]
        ]
        print(createHexGrid(3))
        assert len(expected) == len(createHexGrid(3))

    def testHexBoardWithParameter4(self):
        node = MagicMock()

        expect = [
            [node],
            [node, node],
            [node, node, node],
            [node, node, node, node]
        ]
        assert len(expect) == len(createHexGrid(4))

    def testHexBoardNeighboars(self):

        got = createHexGrid(3)
        print(got[0][0].neighboursDic.values())
        print(got[1])
        assert len(got[0][0].neighboursDic.values()) == 2
        assert list(got[0][0].neighboursDic.values()) == got[1]
        assert got[1][1] in got[1][0].neighboursDic.values()

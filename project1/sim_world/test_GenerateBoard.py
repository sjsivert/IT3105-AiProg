#import pytest
import unittest
from typing import Any
# ofrom mocker import mocker
#from unittest.mock import Mock
from unittest.mock import MagicMixin, MagicMock
from project1.sim_world.Board import HexBoard, BoardState, Boardtype
#from Node import Peg


class HexBoardTest(unittest.TestCase):
    def testHeaxBoardWithSize3(self):
        board = HexBoard(Boardtype.triangle, 3)
        Peg = MagicMock()
        print(board)
        expected = [
            [Peg((0, 0))()],
            [Peg((0, 0))(), Peg((0, 0))()],
            [Peg((0, 0))(), Peg((0, 0))(), Peg((0, 0))()]
        ]
        assert len(expected) == len(board.board)

    def testHexBoardWithParameter4(self):
        board = HexBoard(Boardtype.triangle, 4)
        Peg = MagicMock()

        expect = [
            [Peg()],
            [Peg(), Peg()],
            [Peg(), Peg(), Peg()],
            [Peg(), Peg(), Peg(), Peg()]
        ]
        assert len(expect) == len(board.board)

    def testHexBoardNeighboars(self):

        state = BoardState(
            HexBoard(
                Boardtype.triangle,
                3
            )
        )
        got = state.state
        print(got[0][0].neighboursDic.values())
        print(got[1])
        assert len(got[0][0].neighboursDic.values()) == 2
        assert list(got[0][0].neighboursDic.values()) == got[1]
        assert got[1][1] in got[1][0].neighboursDic.values()

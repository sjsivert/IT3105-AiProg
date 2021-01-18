#import pytest
from typing import Any
import unittest
# ofrom mocker import mocker
#from unittest.mock import Mock
from unittest.mock import MagicMixin, MagicMock
from HexBoard import createHexGrid
from HexBoard import Node


class NodeTest(unittest.TestCase):
    def testCanCreateNode(self):
        # TODO
        assert True
    def testCanAddNeighboar(self):
        node1 = Node()
        node2 = Node()
        node1.addNeighbour(node2, "-1-1")
        assert node2 in node1.neighboursDic.values()

    def testCanNotAddSameNeighbourTwice(self):
        node1 = Node()
        node2 = Node()
        node1.addNeighbour(node2, "-1-1")
        assert node1.addNeighbour(node2, "-1-1") == False


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


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
        node1.addNeighbour(node2)
        assert node2 in node1.neighbours

    def testCanNotAddSameNeighbourTwice(self):
        node1 = Node()
        node2 = Node()
        node1.addNeighbour(node2)
        assert node1.addNeighbour(node2) == False


class HexBoardTest(unittest.TestCase):
    def testHeaxBoardWithSize3(self):
        board = createHexGrid(3)
        node = MagicMock()
        for l in board:
            print(l)
        expected = [
            [Node(1)],
            [Node(2), Node(2)],
            [Node(3), Node(3), Node(3)]
        ]
        for l in expected:
            print(l)
        for l in board:
            print(l)
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
        print(got)
        print(got[0][0].neighbours)
        print(got[1][0].neighbours)
        print(got[2][0].neighbours)
        assert len(got[0][0].neighbours) == 2
        assert got[0][0].neighbours == got[1]
        assert got[1][1] in got[1][0].neighbours


"""  class Test(TestCase):

    def test_create_hex_grid(self):
        node = mock.Mock()
        grid = (createHexGrid(3))
        for list in grid:
            print(grid) """

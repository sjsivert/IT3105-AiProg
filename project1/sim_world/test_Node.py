from unittest.mock import MagicMixin, MagicMock
import unittest
from Node import Node


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

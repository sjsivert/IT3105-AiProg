from unittest.mock import MagicMixin, MagicMock
import unittest
from project1.sim_world.Node import Peg


class PegTest(unittest.TestCase):
    def testCanCreatePeg(self):
        # TODO
        peg = Peg((0, 0))
        assert True

    def testCanAddNeighboar(self):
        peg1 = Peg((0, 0))
        peg2 = Peg((0, 1))
        peg1.addNeighbour(peg2, "-1,-1")
        assert peg2 in peg1.neighboursDic.values()

    def testCanNotAddSameNeighbourTwice(self):
        peg1 = Peg((0, 0))
        peg2 = Peg((0, 1))
        peg1.addNeighbour(peg2, "-1,-1")
        assert peg1.addNeighbour(peg2, "-1,-1") == False

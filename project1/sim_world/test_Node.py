from unittest.mock import MagicMixin, MagicMock
import unittest


class PegTest(unittest.TestCase):
    def testCanCreatePeg(self):
        # TODO
        peg = Peg()
        assert True

    def testCanAddNeighboar(self):
        peg1 = Peg()
        peg2 = Peg()
        peg1.addNeighbour(peg2, "-1,-1")
        assert peg2 in peg1.neighboursDic.values()

    def testCanNotAddSameNeighbourTwice(self):
        peg1 = Peg()
        peg2 = Peg()
        peg1.addNeighbour(peg2, "-1,-1")
        assert peg1.addNeighbour(peg2, "-1,-1") == False

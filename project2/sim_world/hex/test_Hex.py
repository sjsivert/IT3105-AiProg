from unittest.mock import MagicMixin, MagicMock
import unittest
from Hex import Hex


class HexTest(unittest.TestCase):
    def testLoadTournamentState(self):
        # Arrange
        sim_world = Hex(
            boardType="diamond",
            boardWidth=6,
            playerTurn=1,
            loadedHexBoardState=list(range(0, 6*6+1)),
        )

        print(list(range(0, 6*6+1)))
        assert sim_world.convertActionToTournament(2) == (0,1)
        assert sim_world.convertActionToTournament(0) == (0,0)
        assert sim_world.convertActionToTournament(6*6-1) == (5,5)

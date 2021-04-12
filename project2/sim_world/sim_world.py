from typing import List


class SimWorld:
    """
        Interface for SimWorld.
        This class is not finished
    """

    def __init__(self):
        pass

    def getPossibleActions(self) -> List[str]:
        pass

    def isAllowedAction(self, action: str) -> bool:
        pass

    def makeAction(self, action: str):
        pass

    def peekAction(self, action: str):
        pass

    def isWinState(self) -> bool:
        pass

    def getReward(self) -> int:
        pass

    def getPlayerTurn(self) -> int:
        pass

    def getStateHash(self) -> List:
        pass

    def getMaxPossibleActionSpace(self) -> int:
        pass

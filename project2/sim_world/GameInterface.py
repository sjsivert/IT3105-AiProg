class GameInterface:
    def __init__(self):
        pass

    def makeAction(self, action):
        pass

    def isWinState(self) -> bool:
        pass

    def changePlayerTurn(self) -> int:
        pass

    def getStateHash(self) -> str:
        pass

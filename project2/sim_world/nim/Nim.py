from sim_world.GameInterface import GameInterface
from typing import List


class Nim(GameInterface):
    def __init__(
        self,
        numberOfStones,
        maxRemoveEachTurn
    ):
        self.playerTurn = 1
        self.numberofStonesInPile = numberOfStones
        self.maxRemoveEachTurn = maxRemoveEachTurn

    def getPossibleActions(self) -> List[str]:
        return map(str, list(range(1, self.maxRemoveEachTurn + 1)))

    def changePlayerTurn(self) -> int:
        self.playerTurn = 1 if self.playerTurn == 2 else 2
        return self.playerTurn

    def isWinState(self) -> bool:
        return True if self.numberofStonesInPile == 0 else False

    def isAllowedAction(self, action: str) -> bool:
        return action in self.getPossibleActions()

    def makeAction(self, action: str):
        if (not self.isAllowedAction(action)):
            raise("Action not allowed")

        action = int(action)
        if(self.numberofStonesInPile - action < 0):
            raise Exception("Illegal action, not enough stones in pile")
        self.numberofStonesInPile = self.numberofStonesInPile - action

    def playGayme(self):
        while(not self.isWinState()):
            self.changePlayerTurn()
            action = input(
                f"There are {self.numberofStonesInPile} left. \nplayer {self.playerTurn} how many stones do oyou remove?: ")
            self.makeAction(action)

        print(f"Player {self.playerTurn} wins!")

    def __str__(self) -> str:
        return super().__str__(
            f"playerTurn: {self.playerTurn},"
            + f"stonesLeftInPile: {self.numberofStonesInPile}"
        )

    def getStateHash(self) -> str:
        return self.__str__()


if __name__ == "__main__":
    nim = Nim(
        10,
        3
    )
    nim.playGayme()

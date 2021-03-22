from sim_world.sim_world import SimWorld
from typing import List


class Nim(SimWorld):
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
        self.playerTurn = 1 if self.playerTurn == -1 else -1
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

        self.changePlayerTurn()

    def getReward(self) -> int:
        # TODO: Make reward system parameterTunable
        return 10 if self.isWinState else -1

    def getPlayerTurn(self) -> int:
        return self.playerTurn

    def getMaxPossibleActionSpace(self) -> int:
        return self.maxRemoveEachTurn

    def playGayme(self):
        while(not self.isWinState()):
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

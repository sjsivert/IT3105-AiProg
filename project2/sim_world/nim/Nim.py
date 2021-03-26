from sim_world.sim_world import SimWorld
from typing import List


class Nim(SimWorld):
    def __init__(
        self,
        numberOfStones,
        maxRemoveEachTurn
    ):
        self.playerTurn = 1
        self.state = numberOfStones
        self.maxRemoveEachTurn = maxRemoveEachTurn

    def getPossibleActions(self) -> List[int]:
        return list(range(1, self.maxRemoveEachTurn + 1))

    def changePlayerTurn(self) -> int:
        self.playerTurn = 1 if self.playerTurn == -1 else -1
        return self.playerTurn

    def isWinState(self) -> bool:
        return True if self.state == 0 else False

    def isAllowedAction(self, action: int) -> bool:
        return action in self.getPossibleActions()

    def makeAction(self, action: int):
        print(self.getPossibleActions())
        if (not self.isAllowedAction(action)):
            raise("Action not allowed")

        if(self.state - action < 0):
            raise Exception("Illegal action, not enough stones in pile")
        self.state = self.state - action
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
            action = int(action)
            self.makeAction(action)

        print(f"Player {self.playerTurn} wins!")

    def __str__(self) -> str:
        return super().__str__(
            f"playerTurn: {self.playerTurn},"
            + f"stonesLeftInPile: {self.numberofStonesInPile}"
        )

    def getStateHash(self) -> str:
        return [self.playerTurn, self.state]


if __name__ == "__main__":
    nim = Nim(
        10,
        3
    )
    nim.playGayme()

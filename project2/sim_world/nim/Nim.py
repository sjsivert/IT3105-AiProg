from project2.sim_world.sim_world import SimWorld
from typing import List
import math


class Nim(SimWorld):
    def __init__(
        self,
        numberOfStones,
        maxRemoveEachTurn
    ):
        self.maxStones = numberOfStones
        self.playerTurn = 1
        self.state = numberOfStones
        self.maxRemoveEachTurn = maxRemoveEachTurn
        self.actions = list(range(1, maxRemoveEachTurn+1))
    
    def getPossibleActions(self) -> List[int]:
        return list(range(0, min(self.maxRemoveEachTurn, self.state)))

    def changePlayerTurn(self) -> int:
        self.playerTurn = 1 if self.playerTurn == -1 else -1
        return self.playerTurn

    def isWinState(self) -> bool:
        return True if self.state == 0 else False

    def isAllowedAction(self, action: int) -> bool:
        return action in self.getPossibleActions()

    def makeAction(self, action: int):
        #print(" action ", self.actions[action])
        #print(" state ", self.state)
        #print(self.getPossibleActions(), action)
        if self.isWinState():
            return self.getReward()
        if (not self.isAllowedAction(action)):
            raise("Action not allowed")
        if(self.state - self.actions[action] < 0):
            raise Exception("Illegal action, not enough stones in pile")
        self.state = self.state - self.actions[action]
        self.changePlayerTurn()
    
    def peekAction(self, action: int):
        nonstaones = [0] * (self.maxStones - self.state + self.actions[action])
        nonnonstaones = [1] * (self.state - self.actions[action])
        state = [-self.playerTurn] + nonstaones + nonnonstaones 
        return state

    def getReward(self) -> int:
        # TODO: Make reward system parameterTunable
        if self.isWinState():
            # print("reward", 10 * self.playerTurn)
            return 1 * -self.playerTurn

    def getPlayerTurn(self) -> int:
        return self.playerTurn

    def getMaxPossibleActionSpace(self) -> int:
        return self.maxRemoveEachTurn

    def playGayme(self):
        while(not self.isWinState()):
            action = input(
                f"There are {self.state} left. \nplayer {self.playerTurn} how many stones do oyou remove?: ")
            action = int(action)
            self.makeAction(action)

        print(f"Player {self.playerTurn} wins!")

    def playAgainst(self, ANET):
        while(not self.isWinState()):
            action = 0
            if self.playerTurn == -1:
                action = input(
                    f"There are {self.state} left. \nplayer {self.playerTurn} how many stones do oyou remove?: ")
                self.makeAction(int(action) -1)
            else:
                action = ANET.defaultPolicyFindAction(self.getPossibleActions(), self.getStateHash())
                print("There are", self.state, "left. \nplayer", self.playerTurn, " player 2 removes", self.actions[action], "stones")
                self.makeAction(action)

        print(f"Player {self.playerTurn} wins!")

    def __str__(self) -> str:
        return super().__str__(
            f"playerTurn: {self.playerTurn},"
            + f"stonesLeftInPile: {self.numberofStonesInPile}"
        )

    def getStateHash(self) -> str:
        nonstaones = [0] * (self.maxStones - self.state)
        nonnonstaones = [1] * (self.state)
        state = [self.playerTurn] + nonstaones + nonnonstaones 
        return state


if __name__ == "__main__":
    nim = Nim(
        10,
        2
    )
    nim.playGayme()

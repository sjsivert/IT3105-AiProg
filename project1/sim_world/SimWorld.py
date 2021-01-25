from GenerateBoard import Boardtype, BoardState, HexBoard
from typing import List, Tuple


class Action():
    def __init__(self, moveTo: Tuple[int], moveFrom: Tuple[int], jumpOver: Tuple[Tuple]) -> None:
        self.moveFrom = moveFrom
        self.jumpOver = jumpOver
        self.moveTo = moveTo

    def __repr__(self) -> List[Tuple[int]]:
        return [self.moveFrom, self.jumpOver, self.moveTo]

    def __str__(self) -> str:
        return str([self.moveFrom, self.moveTo])


class SimWorld():
    def __init__(
            self,
            boardType: Boardtype,
            boardWith: int,
            removeLocations: List[(int)] = []
    ) -> None:
        hexboard = HexBoard(boardType, boardWith, removeLocations)
        self.boardState = BoardState(hexboard)
        # self.actionListLog
        # self.boardState

    def getLegalActions(self) -> List[Action]:
        state = self.boardState.state
        actionList = []
        # TODO Change to flatmap?
        flatNodeList = []
        for sublist in state:
            for item in sublist:
                flatNodeList.append(item)
        for node in flatNodeList:
            for direction, neighboar in node.neighboursDic.items():
                if neighboar.pegValue == 1 and direction in neighboar.neighboursDic:
                    if neighboar.neighboursDic[direction].pegValue == 0:
                        jumpFrom = node
                        jumpTo = neighboar.neighboursDic[direction]
                        jumpOver = neighboar
                        action = Action(jumpFrom, jumpTo, jumpOver)
                        actionList.append(action)
        return actionList
    # TODO make pegMethod: set value

    def makeAction(self, action) -> bool:
        state = self.boardState.state
        state[action[0][0]][action[0][1]].pegValue = 0
        state[action[1][0]][action[1][1]].pegValue = 0
        state[action[2][0]][action[2][1]].pegValue = 1

    def toHash(self) -> str:
        pass

    def getReward(self) -> int:
        pass

    def stateToHash(self) -> str:
        return str(self.boardState.state)

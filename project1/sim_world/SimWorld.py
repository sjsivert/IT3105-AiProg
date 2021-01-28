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
            boardType: str,
            boardWith: int,
            removeLocations: List[(int)] = []
    ) -> None:
        hexboard = HexBoard(Boardtype[boardType], boardWith, removeLocations)
        self.boardState = BoardState(hexboard)
        self.stateActionLog = []

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

    def getGameLog(self):
        return self.stateActionLog

    def makeAction(self, action) -> bool:
        self.stateActionLog.append((self.boardState, action))
        state = self.boardState
        state.setPegValue((action.moveFrom.location[0], action.moveFrom.location[1]), 0)
        state.setPegValue((action.jumpOver.location[0], action.jumpOver.location[1]), 0)
        state.setPegValue((action.moveTo.location[0], action.moveTo.location[1]), 1)

        return self.getReward()
    
    def peekAction(self, action) -> str:
        state = self.boardState.hexboard.copy()
        state[action.moveFrom.location[0]] [action.moveFrom.location[1]] = 0
        state[action.jumpOver.location[0]][action.jumpOver.location[1]] = 0
        state[action.moveTo.location[0]][action.moveTo.location[1]] = 1
        return str(state)

    def getReward(self):
        if(len(self.getLegalActions()) != 0):
            return -1#TODO generalize
        elif self.boardState.countPegs() == 1:
            return 10#TODO generalize
        return -10#TODO generalize

    def stateToHash(self) -> str:
        return str(self.boardState.state)


from sim_world.GenerateBoard import Boardtype, BoardState, HexBoard
from typing import List, Tuple
import random
from copy import copy, deepcopy


class Action():
    def __init__(self, moveFrom: Tuple[int], jumpOver: Tuple[int], moveTo: Tuple[Tuple]) -> None:
        self.moveFrom = moveFrom
        self.jumpOver = jumpOver
        self.moveTo = moveTo

    def __repr__(self) -> List[Tuple[int]]:
        return [self.moveFrom, self.jumpOver, self.moveTo]

    def __str__(self) -> str:
        return str(self.moveFrom.location[0])+',' + str(self.moveFrom.location[1])+',' + str(self.moveTo.location[0]) + ',' + str(self.moveTo.location[1])


class SAP:
    """
        State Action Pair
    """

    def __init__(self, state, action: Action):
        self.stateHash = state.stateToHash()
        self.state = deepcopy(state._boardState.state)
        self.action = action


class SimWorld():
    def __init__(
            self,
            boardType: str,
            boardWith: int,
            removeLocations: List[(int)] = []
    ) -> None:
        hexboard = HexBoard(Boardtype[boardType], boardWith, removeLocations)
        self.startRemoveLocations = removeLocations
        self._boardState = BoardState(hexboard)
        self._stateActionLog = []

    def getLegalActions(self) -> List[Action]:

        state = self._boardState.state
        if state == None:
            return []
        actionList = []
        # TODO Change to flatmap?
        flatNodeList = []
        for sublist in state:
            for item in sublist:
                flatNodeList.append(item)
        for node in flatNodeList:
            if node.pegValue == 1:
                for direction, neighboar in node.neighboursDic.items():
                    if neighboar.pegValue == 1 and direction in neighboar.neighboursDic:

                        if neighboar.neighboursDic[direction].pegValue == 0:
                            jumpFrom = node
                            jumpTo = neighboar.neighboursDic[direction]
                            jumpOver = neighboar
                            action = Action(jumpFrom, jumpOver, jumpTo)
                            actionList.append(action)
        # for i in actionList:
        #    print(i.moveFrom.location, i.moveTo.location, "f")
        return actionList
    # TODO make pegMethod: set value

    def getState(self) -> List[BoardState]:
        return self._boardState

    def getGameLog(self) -> List[SAP]:
        return self._stateActionLog

    def makeAction(self, action) -> bool:

        self._stateActionLog.insert(
            0, (SAP(self, action)))
        # TODO: Write log to file
        if action == None:
            reward = self.getReward(action)
            #self._boardState.state = None
            return reward
        state = self._boardState
        state.setPegValue(
            (action.moveFrom.location[0], action.moveFrom.location[1]), 0)
        state.setPegValue(
            (action.jumpOver.location[0], action.jumpOver.location[1]), 0)
        state.setPegValue(
            (action.moveTo.location[0], action.moveTo.location[1]), 1)

        return self.getReward(action)

    def getReward(self, action):
        if action != None:
            return -0.01  # TODO generalize
        elif self._boardState.countPegs() == 1:
            return 10  # TODO generalize
        return -10  # TODO generalize

    def stateToHash(self) -> str:
        return str(self._boardState)

    def stateToList(self) -> str:
        return deepcopy(self._boardState.state)

    def RemoveRandomPegs(self, count):
        count = count % self._boardState.countPegs()
        self.startRemoveLocations = []
        while count > 0:
            state = self._boardState
            x = random.randint(0, len(state.state) - 1)
            y = random.randint(0, len(state.state[x]) - 1)
            if state.state[x][y].pegValue == 1:
                state.setPegValue((x, y), 0)
                count -= 1
                self.startRemoveLocations.append((x, y))

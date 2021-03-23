from sim_world.sim_world import SimWorld
from sim_world.hex.Board import Boardtype, BoardState
from sim_world.hex.Board import HexBoard, Boardtype
from sim_world.hex.VisualizeBoard import VisualizePegs
from typing import List, Tuple


class Hex(SimWorld):
    """
        Hex board game
    """

    def __init__(
        self,
        boardType: str,
        boardWith: int,
        playerTurn: int
    ):
        self.playerTurn = playerTurn
        self.boardWidth = boardWith
        hexBoard = HexBoard(Boardtype[boardType], boardWith)
        self.state = BoardState(hexBoard)
        # Dic[actionNumberIndex] -> (x, y) cordinates
        self.possibleActions = self.generatePossibleActions()
        # TODO: Add action log

    def generatePossibleActions(self):
        board = self.state.state
        actions = {}
        count = 0
        for x in range(len(board)):
            for y in range(len(board[x])):
                actions[count] = (x, y)
                count += 1
        return actions

    def getPossibleActions(self) -> List[int]:
        return self.possibleActions.keys()

    def isAllowedAction(self, action: int) -> bool:
        return True if self.getPossibleActions(action) != None else False

    def makeAction(self, action: int):
        action = self.possibleActions[action]
        self.state.setPegValue(action, self.playerTurn)  # Update boardState
        # remove action from possibleActions
        self.possibleActions[action] = None
        self.changePlayerTurn()

    def isWinState(self) -> bool:
        k = 4 - 1
        pass

    def generateBoardSideCordinates(self):
        #upperLeft = [00, 10, 20]
        upperRight = [11, 22, 33]
        lowerLeft = [30, 40, 50]
        lowerRight = [42, 51, 60]

        upperLeft = [(n, 0) for n in range(self.boardWidth - 1)]
        upperRight = [(x, x) for x in range(1, self.boardWidth)]
        lowerLeft = [(x, 0)
                     for x in range(self.boardWidth - 1, self.boardWidth*2 - 2)]
        # lowerRight = [(x, 0) for]  # TODO

        print(upperLeft)
        print(upperRight)
        print(lowerLeft)
        return upperLeft, upperRight, lowerLeft, lowerRight

    def depthFirstSearch(self, node, isTheRightState):
        # if isTheRightState
        #  return True
        # for child in children:
        # depthFirstSearch(child)
        pass

    def getReward(self) -> int:
        pass

    def changePlayerTurn(self) -> int:
        self.playerTurn = 1 if self.playerTurn == -1 else -1
        return self.playerTurn

    def getStateHash(self) -> str:
        pass

    def getMaxPossibleActionSpace(self) -> int:
        return self.boardWidth**2

    def visualizeBord(self):
        VisualizePegs(
            pegList=self.state
        )

    def playGame(self):
        self.generateBoardSideCordinates()
        while (not self.isWinState()):
            self.changePlayerTurn()
            self.visualizeBord()
            action = input(
                f"Player {self.playerTurn}, where do you place your peg? "
            )
            self.makeAction(action)


class Cordinates:
    def __init__(self, x, y):
        self.x = x
        self.y = y

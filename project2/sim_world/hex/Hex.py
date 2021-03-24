from sim_world.sim_world import SimWorld
from sim_world.hex.Board import Boardtype, BoardState
from sim_world.hex.Node import Peg
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
        self.lastAction = None
        hexBoard = HexBoard(Boardtype[boardType], boardWith)
        self.state = BoardState(hexBoard)
        # Dic[actionNumberIndex] -> (x, y) cordinates
        self.possibleActions = self.generatePossibleActions()

        self.upperLeft, self.upperRight, self.lowerLeft, self.lowerRight = self.generateBoardSideCordinates()
        # TODO: Add action log
        self.generateTournamentActionMaps()

    def generateTournamentActionMaps(self):  # {simworld action (row, column): tournament acition (row, column)}
        self.simWorldToTournament = {}
        for c in range(len(board)):
            for r in range(len(board[c])):
                col = c
                row = r
                if(col < boardSize):
                    self.simWorldToTournament[(c,r)] = (col-row,row)
                else:
                    self.simWorldToTournament[(c,r)] = (self.boardWidth-row-1,row+col-self.boardWidth+1)
        self.tournamentToSimworld = {}
        for key in self.simWorldToTournament.keys():
            self.tournamentToSimworld[self.simWorldToTournament[key]] = key

    def tournamentStateToSimworldState(self, state) -> list:  # state is list [player, 0,0,1,2,0,...] -> [[0,0],[1,0],[1,1]] (player missing)
        simWorldBoard = []
        for i in range(1, self.boardWidth+1):
            simWorldBoard.append(i * [0])
        for i in range(boardWidth - 1, 0, -1):
            simWorldBoard.append(i * [0])  # Creates empty sim worl board
        for col in range(len(simWorldBoard)):
            for row in range(len(simWorldBoard[col])):
                if(col < boardSize):
                    self.simWorldToTournament[(col,row)] = (col-row,row)
                else:
                    self.simWorldToTournament[(col,row)] = (self.boardWidth-row-1,row+col-self.boardWidth+1)
        for col in range(self.boardWidth):
            for row in range(self.boardWidth):
                coord = self.tournamentToSimworld[(col, row)]
                simWorldBoard[coord[0]][coord[1]] = state[col*self.boardWidth+row+1]
        return simWorldBoard
        
    def generatePossibleActions(self):
        board = self.state.state
        actions = {}
        count = 0
        for x in range(len(board)):
            for y in range(len(board[x])):
                actions[count] = (x, y)
                count += 1
        return actions

    def getPossibleActions(self):
        return self.possibleActions.keys()

    def isAllowedAction(self, actionTuple: Tuple[int]) -> bool:
        for key in self.getPossibleActions():
            value = self.possibleActions.get(key)
            if value == actionTuple:
                return key
        return False
        # return True if action in self.possibleActions.values() else False

    def makeAction(self, action: int):
        """
            Makes and action, changes state.
            And changes playerTurn.
        """
        print(f"Action: {action}")
        actionTuple = self.possibleActions.get(action)
        self.state.setPegValue(
            actionTuple, self.playerTurn)  # Update boardState
        # remove action from possibleActions
        self.possibleActions[action] = None
        self.lastAction = self.state.getPegNode(actionTuple)
        print(type(self.lastAction))
        self.changePlayerTurn()

    def isWinState(self) -> bool:
        if self.playerTurn == -1 and self.lastAction:
            return self.depthFirstSearch(
                node=self.lastAction,
                visited=[],
                startLocations=self.upperLeft,
                endLocations=self.lowerRight
            )
        elif self.playerTurn == 1 and self.lastAction:
            return self.depthFirstSearch(
                node=self.lastAction,
                visited=[],
                startLocations=self.upperRight,
                endLocations=self.lowerLeft
            )

    def depthFirstSearch(
        self,
        node: Peg,
        visited: List,
        startLocations: List,
        endLocations: List
    ):
        print(f"node: {node.location}")
        print(node.getChildren())
        if node not in visited:
            print(f"Visiting: {node.location}")
            visited.append(node)

            # Check if winState
            visitedLocationTuples = list(
                map(lambda peg: peg.location, visited))
            print(f"sets: {set(startLocations)}")
            print(f"sets: {set(visitedLocationTuples)}")
            print(
                f"Have visited startLocation? {set(visitedLocationTuples) & set(startLocations)}")
            print(
                f"Have visited endLocation: {set(endLocations) & set(visitedLocationTuples) }")
            haveVisitedAEndLocationAndhaveVisitedAStartLocation = (
                set(startLocations) & set(visitedLocationTuples)) and (set(endLocations) & set(visitedLocationTuples))
            if (haveVisitedAEndLocationAndhaveVisitedAStartLocation):
                print("Win state reached!")
                return True
            for child in node.getChildren():
                if child.pegValue == node.pegValue:
                    print(f"About to visit child: {child.location}")
                    if(self.depthFirstSearch(
                        node=child,
                        visited=visited,
                        startLocations=startLocations,
                        endLocations=endLocations
                    )):
                        return True
        return False

    def generateBoardSideCordinates(self):
        """
            The sides of the board used to check if a win state is reached.
            Player 1 must have a connected path from upperLeft location to a lowerRight location.
            Player 2 must have the same for upperRight, to lowerLeft
            Example locations of a board with width 3:
                upperLeft = [00, 10, 20]
                upperRight = [11, 22, 33]
                lowerLeft = [30, 40, 50]
                lowerRight = [42, 51, 60]
        """
        upperLeft = [(n, 0) for n in range(self.boardWidth - 1)]
        upperRight = [(x, x) for x in range(1, self.boardWidth)]
        lowerLeft = [(x, 0)
                     for x in range(self.boardWidth - 1, self.boardWidth*2 - 2)]
        lowerRight = [(x, y) for x, y in zip(range(
            self.boardWidth, self.boardWidth*2 - 1), range(self.boardWidth - 2, -1, -1))]

        return upperLeft, upperRight, lowerLeft, lowerRight

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
            print(self.possibleActions)
            self.visualizeBord()
            playerInput = input(
                f"Player {self.playerTurn}, where do you place your peg? "
            )
            actionTuple = tuple(list(map(int, playerInput.split(','))))
            print(actionTuple)
            actionNumber = self.isAllowedAction(actionTuple)
            print(actionNumber)
            if(actionNumber == False and actionNumber != 0):
                raise Exception("Not a valid action")
            self.makeAction(actionNumber)

        # Since changePLayer is done in makeAction(), the winner is the opposite of current player
        print(f"Player {self.playerTurn * -1} wins the game!!")
        # Visualise board one last time to get end result
        self.visualizeBord()


class Cordinates:
    def __init__(self, x, y):
        self.x = x
        self.y = y

from copy import deepcopy

"""
Board in the diamond form. This is board of size 4, with pin numbers
      0
    4   1
  8   5   2
12  9   6   3
  13  10  7
    14  11
      15

Representation:
[ 0, 1, 2, 3],
[ 4, 5, 6, 7],
[ 8, 9,10,11],
[12,13,14,15]
"""


class DiamondBoard:

    def __init__(self, emptyStartPins, size):
        self.emptyStartPins = emptyStartPins
        self.size = size
        self.gameBoard = []
        self.legalMoveDirections = [[-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0]]
        self.legalChildStates = []
        self.resetBoard()
        self.printGameBoard()

    def resetBoard(self):
        self.gameBoard = []
        for c in range(self.size):
            self.gameBoard.append([1] * self.size)
            for r in range(self.size):
                if int(c * self.size + r) in self.emptyStartPins:
                    self.gameBoard[c][r] = 0
        self.updateLegalChildStates()
        #self.printGameBoard()

    def printGameBoard(self):
        print("------------")
        for row in self.gameBoard:
            print(" "*len(row) + str(row))

    def legalMoveLeft(self):
        self.updateLegalChildStates()
        return len(self.legalChildStates) > 0

    def updateLegalChildStates(self):
        self.legalChildStates = []
        for r in range(self.size):
            for c in range(self.size):
                if self.gameBoard[r][c] == 0:
                    for move in self.legalMoveDirections:
                        rowStart = r + 2 * move[0]
                        colStart = c + 2 * move[1]
                        if 0 <= rowStart < self.size and 0 <= colStart < self.size:
                            if self.gameBoard[rowStart][colStart] == 1:
                                if self.gameBoard[r + move[0]][c + move[1]] == 1:
                                    childState = deepcopy(self.gameBoard)
                                    childState[rowStart][colStart] = 0  # Jump start point
                                    childState[r + move[0]][c + move[1]] = 0
                                    childState[r][c] = 1  # Jump end point
                                    self.legalChildStates.append(childState)

    def getLegalChildStates(self):
        self.updateLegalChildStates()
        return self.stringifyStates(self.legalChildStates)

    def stringifyStates(self, states):
        stringStates = []
        for state in states:
            stringStates.append(self.stringifyState(state))
        return stringStates

    def stringifyState(self, state):
        stringState = ""
        for row in state:
            for pin in row:
                stringState += str(pin)
        return stringState

    def getCurrentStringState(self):
        return self.stringifyState(self.gameBoard)

    def executeStringState(self, stringState):
        gameBoard = []
        for c in range(self.size):
            row = []
            for r in range(self.size):
                row.append(int(stringState[c * self.size + r]))
            gameBoard.append(row)
        if gameBoard in self.legalChildStates:
            self.gameBoard = deepcopy(gameBoard)
        else:
            print("Illegal move")
        self.updateLegalChildStates()

    def getReinforcement(self):
        if self.legalMoveLeft():
            return 0
        else:
            pinsLeft = self.getPinsLeft()
            if pinsLeft == 1:
                return 100
            else:
                return -10

    def getPinsLeft(self):
        pinNum = 0
        for row in self.gameBoard:
            for pin in row:
                if pin == 1:
                    pinNum += 1
        return pinNum
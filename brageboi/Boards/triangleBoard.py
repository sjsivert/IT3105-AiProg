from copy import deepcopy

"""
Board in the diamond form. This is board of size 4, with pin numbers
      0
    1   2
  3   4   5
6   7   8   9

Representation:
[0],
[1, 2],
[3, 4, 5]
[6, 7, 8, 9]
"""


class TriangleBoard:

    def __init__(self, emptyStartPins=None, size=None):
        self.emptyStartPins = emptyStartPins
        if self.emptyStartPins is None:
            self.emptyStartPins = [0]
        self.size = size
        if self.size is None:
            self.size = 5
        self.gameBoard = []
        self.legalMoveDirections = [[-1, -1], [-1, 0], [0, -1], [0, 1], [1, 0], [1, 1]]
        self.legalChildStates = []
        self.resetBoard()
        self.printGameBoard()

    def resetBoard(self):
        self.gameBoard = []
        rowLen = 1
        for r in range(self.size):
            self.gameBoard.append([1] * rowLen)
            for c in range(rowLen):
                if int(r * rowLen/2 + c) in self.emptyStartPins:
                    self.gameBoard[r][c] = 0
            rowLen += 1
        self.updateLegalChildStates()

    def printGameBoard(self):
        print("---Board----")
        for row in self.gameBoard:
            print(" "*(self.size - len(row)) + str(row))

    def legalMoveLeft(self):
        self.updateLegalChildStates()
        return len(self.legalChildStates) > 0

    def updateLegalChildStates(self):
        self.legalChildStates = []
        for r in range(len(self.gameBoard)):
            for c in range(len(self.gameBoard[r])):
                if self.gameBoard[r][c] == 0:
                    for move in self.legalMoveDirections:
                        rowStart = r + 2 * move[0]
                        colStart = c + 2 * move[1]
                        if 0 <= rowStart < self.size and 0 <= colStart < len(self.gameBoard[r])+2:
                            if colStart <= rowStart:
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
        rowLen = 1
        for r in range(self.size):
            row = []
            for c in range(rowLen):
                row.append(int(stringState[int(r * rowLen/2 + c)]))
            gameBoard.append(row)
            rowLen += 1
        if gameBoard in self.legalChildStates:
            self.gameBoard = deepcopy(gameBoard)
        else:
            print("Illegal move")
        self.legalChildStates = []

    def getReinforcement(self):
        if self.legalMoveLeft():
            return 0
        else:
            if self.getPinsLeft() == 1:
                return 500
            else:
                return -10

    def getPinsLeft(self):
        pinNum = 0
        for row in self.gameBoard:
            for pin in row:
                if pin == 1:
                    pinNum += 1
        return pinNum


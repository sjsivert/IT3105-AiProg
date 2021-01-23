import networkx as nx
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
import plotly.basedatatypes as base
import plotly.graph_objects as go
from Node import Peg
from enum import Enum


class Boardtype(Enum):
    triangle = "triangle"
    diamond = "diamond"


class HexBoard():
    def __init__(self, boardType: Boardtype, boardWidth: int, removeLocations: list[tuple[int]] = []) -> None:
        if (boardType is Boardtype.triangle):
            self.board = self.generateTriangle(width=boardWidth)
            self.board = self.removePegs(removeLocations, self.board)
        elif (boardType is Boardtype.diamond):
            self.board = self.generateDiamond(width=boardWidth)
            self.board = self.removePegs(removeLocations, self.board)
        else:  # boardType has to be diamond
            raise ValueError("BoardType has to be triangle or diamond")

    def generateTriangle(self, width: int) -> list[list[int]]:
        board = []
        for i in range(1, width+1):
            board.append(i * [1])
        return board

    def generateDiamond(self, width: int) -> list[list[int]]:
        board = self.generateTriangle(width)
        for i in range(width - 1, 0, -1):
            board.append(i * [1])
        return board

    def removePegs(self, removeLocations: list[tuple[int]], board: list[list[int]]) -> list[list[int]]:
        for removeLocation in removeLocations:
            board[removeLocation[0]][removeLocation[1]] = 0
        return board

    def __repr__(self):
        return self.board

    def __str__(self):
        return str(self.board)

    def __iter__(self):
        return self.board


class BoardState():
    def __init__(self, hexBoard: HexBoard) -> None:
        self.state = self.boardToNodes(hexBoard=hexBoard)

    def boardToNodes(self, hexBoard: HexBoard) -> list[list[Peg]]:
        board = hexBoard.board
        pegList = []
        for x in range(len(board)):
            layerList = []
            for y in range(len(board[x])):
                layerList.append(Peg((x, y), board[x][y]))
                if(y > 0):
                    layerList[y].addBiDirectionalNeighbour(
                        layerList[y - 1], "0,-1")
                if(x > 0):
                    layerGrowthFactor = len(board[x - 1]) - len(board[x])
                    secoundParentConnection = y + layerGrowthFactor  # other node above you
                    if(y < len(pegList[x-1])):
                        # add yourself as connection to node above you
                        pegList[x-1][y].addBiDirectionalNeighbour(
                            layerList[y], "-1," + str(-layerGrowthFactor))
                    if(secoundParentConnection >= 0 and secoundParentConnection < len(pegList[x - 1])):
                        # add yourself as connection to other node above you
                        pegList[x-1][secoundParentConnection].addBiDirectionalNeighbour(
                            layerList[y], "-1," + str(layerGrowthFactor))

            pegList.append(layerList)
        return pegList

    def __repr__(self):
        return self.state

    def __str__(self):
        return str(self.state)


if __name__ == "__main__":
    print(HexBoard(Boardtype["triangle"], 4))
    board = HexBoard(Boardtype["diamond"], 4)
    state = BoardState(board)
    print(state)
    # print(GenerateDiamond(4))
    # b = GenerateTriangle(4)
    # print(removePegs([(0, 0), (1, 0)], b))
    # pegList = BoardToNodes(b)

    """def plotTree(node: Peg, G, drawnList: list) -> None:
        print(node.location)
        if node not in drawnList:
            G.add_node(node.location)
            drawnList.append(node)
            for child in node.neighboursDic.values():
                G.add_edge(node.location, child.location)
                plotTree(child, G, drawnList)
        return G
        
    def draw_figure(rootNode: Peg):
        G = nx.Graph()
        G = plotTree(rootNode, G, [])

        plt.subplot(121)
        nx.draw(
            G,
            with_labels=True,
            #node_color={"blue", "red"},
            font_weight='bold')
        plt.show()
        plt.savefig("grid.png")

    draw_figure(
        pegList[3][0]
    )"""

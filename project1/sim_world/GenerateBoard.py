import json
from Node import Peg
import networkx as nx

import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
import plotly.basedatatypes as base
from Node import Node
import plotly.graph_objects as go


def GenerateTriangle(width):
    board = []
    for i in range(1, width+1):
        board.append(i * [1])
    return board

def GenerateDiamond(width):
    board = GenerateTriangle(width)
    for i in range(width - 1, 0, -1):
        board.append(i * [1])
    return board

def removePegs(removeLocations, board):
    for removeLocation in removeLocations:
        board[removeLocation[0]]  [removeLocation[1]] = 0
    return board


def BoardToNodes(board):
    pegList = []
    for x in range(len(board)):
        layerList =[]
        for y in range(len(board[x])):
            layerList.append(Peg(  (x, y),board[x][y] ) )
            if(y > 0):
                layerList[y].addBiDirectionalNeighbour(layerList[y -1], "0,-1")
            if(x > 0):
                layerGrowthFactor = len(board[x -1]) - len(board[x])
                secoundParentConnection = y + layerGrowthFactor#other node above you
                if(y < len(pegList[x-1])):
                    pegList[x-1][y].addBiDirectionalNeighbour(layerList[y], "-1," + str(-layerGrowthFactor)) #add yourself as connection to node above you
                if(secoundParentConnection >= 0 and secoundParentConnection < len(pegList[x -1])):
                    pegList[x-1][secoundParentConnection].addBiDirectionalNeighbour(layerList[y], "-1,"+ str(layerGrowthFactor)) #add yourself as connection to other node above you

        pegList.append(layerList)
    return pegList

print (GenerateTriangle(4))
print (GenerateDiamond(4))
b = GenerateTriangle(4)
print (removePegs([(0,0), (1,0)], b))
pegList = BoardToNodes(b)     

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

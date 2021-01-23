import json
from Node import Peg
import networkx as nx

import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
import plotly.basedatatypes as base
from Node import Node
import plotly.graph_objects as go

def plotTree(node: Peg, G, drawnList: list) -> None:
    print(node.location)
    if node not in drawnList:
        G.add_node(node.location)
        drawnList.append(node)
        for child in node.neighboursDic.values():
            G.add_edge(node.location, child.location)
            plotTree(child, G, drawnList)
    return G


def VisualizePegs(pegList, lastAction =[], pegColor ='#0000ff',noPegColor ='#000000',nodeSize = 120, movedSize = 300):
    nodes = []
    colors = []
    pos ={}
    nodeSizes = []
    for layer in range(len(pegList)):
        for node in range(len(pegList[layer])):
            
            nodes.append(pegList[layer][node].location)

            halfLayerSize = (len(pegList[layer]) / 2)
            nodeXPos = node - halfLayerSize
            nodeYPos =  0.9 - (layer / len(pegList)) * 1.8
            pos[pegList[layer][node].location] = [nodeXPos, nodeYPos]

            colors.append(noPegColor)
            if pegList[layer][node].pegValue == 1:
                colors[-1] = pegColor

            nodeSizes.append(nodeSize)
            if pegList[layer][node].location in lastAction and lastAction[1] != pegList[layer][node].location:
                nodeSizes[-1] = movedSize


    G = nx.Graph()
    G = plotTree(pegList[0][0], G, [])

    plt.subplot(121)
    nx.draw_networkx(G, pos, nodelist=nodes, node_size=nodeSizes, node_color=colors,with_labels=False)
    plt.ylim(-1,1)
    plt.axis('off')
    plt.show()

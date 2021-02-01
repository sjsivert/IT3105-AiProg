import json
from Node import Peg
import networkx as nx

import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
import plotly.basedatatypes as base
from Node import Node
import plotly.graph_objects as go
import cv2
import numpy as np

videoFps = 2

# TODO add to class


def plotTree(node: Peg, G, drawnList: list) -> None:
    if node not in drawnList:
        G.add_node(node.location)
        drawnList.append(node)
        for child in node.neighboursDic.values():
            G.add_edge(node.location, child.location)
            plotTree(child, G, drawnList)
    return G


def VisualizePegs(pegList, stepNumber=0, lastAction=None, pegColor='#0000ff', noPegColor='#000000', nodeSize=120, movedSize=300):
    nodes = []
    colors = []
    pos = {}
    nodeSizes = []
    pegList = pegList.getStateList()
    for layer in range(len(pegList)):
        for node in range(len(pegList[layer])):

            nodes.append(pegList[layer][node].location)

            halfLayerSize = (len(pegList[layer]) / 2)
            nodeXPos = node - halfLayerSize
            nodeYPos = 0.9 - (layer / len(pegList)) * 1.8
            pos[pegList[layer][node].location] = [nodeXPos, nodeYPos]

            colors.append(noPegColor)
            if pegList[layer][node].pegValue == 1:
                colors[-1] = pegColor

            nodeSizes.append(nodeSize)
            if lastAction != None:
                if pegList[layer][node].location == lastAction.moveFrom.location:
                    nodeSizes[-1] = movedSize
                if pegList[layer][node].location == lastAction.moveTo.location:
                    nodeSizes[-1] = movedSize

    G = nx.Graph()
    G = plotTree(pegList[0][0], G, [])

    fig, ax = plt.subplots()

    nx.draw_networkx(G, pos, nodelist=nodes, node_size=nodeSizes,
                     node_color=colors, with_labels=False)
    plt.ylim(-1, 1)
    ax.set_facecolor('deepskyblue')
    plt.axis('off')
    fig.set_facecolor('deepskyblue')
    plt.savefig('images/' + str(stepNumber) + '.png')
    plt.close()


def GenerateVideo(stepNumber, name):
    img_array = []
    for filename in range(stepNumber + 1):
        img = cv2.imread('images/' + str(filename) + '.png')
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    out = cv2.VideoWriter(str(name) + '.avi',
                          cv2.VideoWriter_fourcc(*'DIVX'), videoFps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

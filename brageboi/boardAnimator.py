import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class BoardAnimator:

    def __init__(self, frameDelay):
        self.frameDelay = frameDelay
        self.G = nx.Graph()
        self.nodes = {}
        self.edges = []
        self.colorProgression = []
        self.line = None

    def animateEpisode(self, usingDiamondBoard, boardSize, episodeStates):
        if usingDiamondBoard:
            self.buildDiamondBoard(boardSize)
            fig, ax = plt.subplots(figsize=(boardSize, 1.7*boardSize))
        else:
            self.buildTriangleBoard(boardSize)
            fig, ax = plt.subplots(figsize=(boardSize, boardSize))
        self.colorProgression = []
        for state in episodeStates:
            self.colorProgression.append(self.getFrameColors(state))
        anim = FuncAnimation(fig, self.animate, interval=self.frameDelay, frames=len(episodeStates), repeat=False)
        plt.draw()
        plt.show()

    def buildDiamondBoard(self, boardSize):
        self.nodes = {}
        nodeNum = 0
        for row in range(boardSize):
            for col in range(boardSize):
                xCoord = (col - row + int(boardSize/2))
                yCoord = boardSize - col - row
                self.nodes[nodeNum] = (xCoord, yCoord)
                nodeNum += 1

    def buildTriangleBoard(self, boardSize):
        self.nodes = {}
        nodeNum = 0
        rowLen = 0
        for row in range(boardSize):
            rowLen += 1
            for col in range(rowLen):
                xCoord = (2*col - row) + 2*boardSize
                yCoord = boardSize - row
                self.nodes[nodeNum] = (xCoord, yCoord)
                nodeNum += 1

    def getFrameColors(self, state):
        nodeColors = []
        for pin in state:
            if pin == "0":
                nodeColors.append("black")
            else:
                nodeColors.append("lime")
        return nodeColors

    def animate(self, frame):
        self.G.add_nodes_from(self.nodes.keys())
        nx.draw(self.G, pos=self.nodes, node_color=self.colorProgression[frame], with_labels=True)

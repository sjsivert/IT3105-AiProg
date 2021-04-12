import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class BoardAnimator:

    def __init__(self, boardSize, frameDelay=500):
        self.frameDelay = frameDelay
        self.G = nx.Graph()
        self.nodes = {}
        self.edges = []
        self.colorProgression = []
        self.line = None
        self.animationStates = []
        self.colorProgression = []
        self.boardSize = boardSize
        
    def addAnimationState(self, animationState): # Tar inn en state list.
        self.animationStates.append(animationState[1:])
        self.colorProgression.append(self.getFrameColors(animationState[1:]))
        

    def animateEpisode(self):
        self.buildBoard()
        fig, ax = plt.subplots(figsize=(1.33*self.boardSize, 2*self.boardSize))
        anim = FuncAnimation(fig, self.animate, interval=self.frameDelay, frames=len(self.animationStates), repeat=False)
        plt.draw()
        plt.show()

    def buildBoard(self):
        self.nodes = {}
        nodeNum = 0
        rowLen = 1
        for row in range(self.boardSize):
            for col in range(rowLen):
                xCoord = self.boardSize + col*2 - rowLen
                yCoord = row
                self.nodes[nodeNum] = (xCoord, yCoord)
                nodeNum += 1
            rowLen += 1
        rowLen -= 2
        while rowLen >= 1:
            for col in range(rowLen):
                xCoord = self.boardSize + col*2 - rowLen
                yCoord = 2*self.boardSize - rowLen-1
                self.nodes[nodeNum] = (xCoord, yCoord)
                nodeNum += 1
            rowLen -= 1
        print("nodes", self.nodes)

    def getFrameColors(self, state):
        nodeColors = []
        for pin in state:
            if pin == 0:
                nodeColors.append("black")
            elif pin == -1:
                nodeColors.append("lime")
            elif pin == 1:
                nodeColors.append("red")
            else:
                print("Hva faen ", pin)
        return nodeColors

    def animate(self, frame):
        self.G.add_nodes_from(self.nodes.keys())
        nx.draw(self.G, pos=self.nodes, node_color=self.colorProgression[frame], with_labels=True)


if __name__ == '__main__':
    BA = BoardAnimator(boardSize = 3)
    BA.addAnimationState([-1,0,0,0,0,0,0,0,0,0])
    BA.addAnimationState([1,0,0,0,-1,0,0,0,0,0])
    BA.addAnimationState([-1,0,0,0,-1,0,1,0,0,0])
    BA.addAnimationState([1,0,0,0,-1,0,1,0,-1,0])
    BA.animateEpisode()

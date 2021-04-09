import math

class TreeNode:
    def __init__(self, state, possibleActions:int, parent=None):
        self.numTimesVisited = 1
        self.numTakenAction = [0.00001] * possibleActions
        self.totalEvaluation = 0  # Accumulated  evaluation of this node
        self.children = {}
        self.parent = parent
        self.state = state
        self.c = 0.5

    def getExpectedResult(self, action: int) -> float:
        #print(self.children[action].totalEvaluation, self.numTakenAction[action])
        return self.children[action].totalEvaluation / self.numTakenAction[action]

    def addChild(self, action: int, child) -> None:
        if action in self.children.keys():
            raise Exception("duplicate child is illigal (no twins!)")
        self.children[action] = child
        child.parent = self
        return child
        
    def getExplorationBias(self, action: int) -> float:
        return self.c * math.sqrt(math.log(self.numTimesVisited) / self.numTakenAction[action])

    def addActionTaken(self, action: int):
        self.numTakenAction[action] += 1
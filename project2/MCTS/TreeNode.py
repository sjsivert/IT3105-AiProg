import math


class TreeNode:
    def __init__(self, state, parent=None):
        self.numTimesVisited = 0,
        self.numTakenAction = {}
        self.totalEvaluation = 0  # Accumulated  evaluation of this node
        self.children = {}
        self.parent = parent
        self.state = state
        self.c = 1

    def getExpectedResult(self, action: str) -> float:
        if action not in self.numTakenAction:
            return 0
        return self.totalEvaluation / self.numTakenAction.get(action)

    def addChild(self, action: str, child: TreeNode) -> None:
        if action in self.children.keys():
            raise Exception("duplicate child is illigal (no twins!)")
        self.children[action] = child

    def getExplorationBias(self, action: str) -> float:
        return self.c * math.sqrt(math.log(self.numTimesVisited) / self.numTakenAction.get(action, 0))

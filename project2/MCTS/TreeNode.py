class TreeNode:
    def __init__(self):
        self.numTimesVisited = 0,
        self.numTakenAction = {}
        self.totalEvaluation = 0  # Accumulated  evaluation of this node

    def getQValue(self, action: str) -> int:
        return self.totalEvaluation / self.numTakenAction.get(action)

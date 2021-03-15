from TreeNode import TreeNode
from sim_world import SimWorld
class MCTS:
    def __init__(
        self, 
        root: TreeNode
        ):
        self.simWorld = SimWorld(root.state)
        self.currentNode = root

    def treePolicyFindAction(self, opponentFactor:int)-> str:
        # TODO: Check if this works as expected
        bestAction = None
        bestValue = -1000000
        for action in self.root.numTakenAction.key():
            currentActionNodeValue = ( opponentFactor * self.currentNode.getExpectedResult(action) ) + self.currentNode.getExplorationBias(action) 
            if(currentActionNodeValue > bestValue):
                bestValue = currentActionNodeValue
                bestAction = action
        return action

    

    def treeSearch(self) -> TreeNode:
        while len(self.currentNode.numTakenAction) != 0:
            opponentFactor = (1 -(i % 2) * 2)
            treePolicyAction = self.treePolicyFindAction(opponentFactor)
            currentNode = self.currentNode.children.get(treePolicyAction)
            self.simWorld.makeAction(treePolicyAction)
        return currentNode

"""
    def doGames(self, rolloutsPerLeaf:int, numberOfTreeGames:int, numberOfGames:int)-> None:
        for i in range(numberOfGames):
            self.simWorld = SimWorld()
            currentState = self.simWorld.__str__()
            self.root = TreeNode(state = currentState, parent = None) 
            while not self.simWorld.isWinState():
                monteCarloSimWorld = SimWorld(currentState)
                for i in range(numberOfTreeGames):
                    pass

"""




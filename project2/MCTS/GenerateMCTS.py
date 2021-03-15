import math
from TreeNode import TreeNode
from sim_world import SimWorld


class MCTS:
    def __init__(
        self,
        root: TreeNode
    ):
       # self.simWorld = SimWorld(root.state)
        self.rootNode = root
        self.currentNode = root
        self.currentLeafNode = root

    def treePolicyFindAction(self, opponentFactor: int) -> str:
        # TODO: Check if this works as expected
        # Or if we should use max() and min()
        bestAction = None
        bestValue = -math.inf()
        for action in self.root.numTakenAction.key():
            currentActionNodeValue = (opponentFactor * self.currentNode.getExpectedResult(
                action)) + self.currentNode.getExplorationBias(action)
            if(currentActionNodeValue > bestValue):
                bestValue = currentActionNodeValue
                bestAction = action
        return action

    def nodeExpansion(self):
        possibleActions = self.simWorld.getPossibleActions()
        for action in possibleActions:
            self.currentNode.addChild(
                action=action,
                child=TreeNode(self.simWorld.state)
            )

    def treeSearch(self, node: TreeNode, playerTurn: int) -> TreeNode:
        self.simWorld = SimWorld(node.state, playerTurn)

        while len(self.currentNode.numTakenAction) != 0:
            opponentFactor = self.simWorld.playerTurn
            treePolicyAction = self.treePolicyFindAction(opponentFactor)
            currentNode = self.currentNode.children.get(treePolicyAction)

            self.simWorld.makeAction(treePolicyAction)
            self.simWorld.changePlayerTurn()
        self.currentLeafNode = currentNode
        return currentNode

    def rollout(self):
        pass


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

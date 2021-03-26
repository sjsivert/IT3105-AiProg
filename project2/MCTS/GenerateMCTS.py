import math
import copy
from MCTS.TreeNode import TreeNode
from sim_world.sim_world import SimWorld


class MCTS:
    def __init__(
        self,
        root: TreeNode
    ):
       # self.simWorld = SimWorld(root.state)
        self.rootNode = root
        self.currentNode = root
        self.currentLeafNode = root

    def treePolicyFindAction(self, opponentFactor: int) -> int:
        # TODO: Check if this works as expected
        # Or if we should use max() and min()
        bestAction = None
        bestValue = -math.inf
        for action in range(len(self.currentNode.numTakenAction)):
            currentActionNodeValue = (opponentFactor * self.currentNode.getExpectedResult(
                action)) + self.currentNode.getExplorationBias(action)
            if(currentActionNodeValue > bestValue):
                bestValue = currentActionNodeValue
                bestAction = action
        return bestAction

    def nodeExpansion(self):
        possibleActions = self.simWorld.getPossibleActions()
        for action in possibleActions:
            self.currentNode.addChild(
                action=action,
                child=TreeNode(self.simWorld.state,
                               self.simWorld.getMaxPossibleActionSpace())
            )

    def makeAction(self, action: int):
        self.currentNode.addActionTaken(action)

        self.currentNode = self.currentNode.children.get(action)
        self.simWorld.makeAction(action)

        return self.currentNode

    def makeSearchAction(self, action: int):
        self.currentNode.addActionTaken(action)
        self.currentNode = self.currentNode.children.get(action)
        self.simWorld.makeAction(action)

        self.currentNode.numTimesVisited += 1
        return self.currentNode

    def treeSearch(self, node: TreeNode, simWorld) -> TreeNode:
        self.simWorld = copy.deepcopy(simWorld)
        currentNode = self.currentNode
        while len(self.currentNode.children) != 0:
            print(currentNode)
            playerTurn = self.simWorld.playerTurn
            treePolicyAction = self.treePolicyFindAction(playerTurn)
            currentNode = self.makeSearchAction(treePolicyAction)
        self.currentLeafNode = currentNode
        self.nodeExpansion()

        return currentNode

    def rollout(self, ANET):
        while not self.simWorld.isWinState():
            defaultPolicyAction = self.defaultPolicyFindAction(ANET)
            self.simWorld.makeAction(defaultPolicyAction)
        return self.simWorld.getReward()

    def defaultPolicyFindAction(self, ANET) -> int:
        distribution  = ANET.getDistributionForState(self.simWorld.getStateHash())
        print(distribution)
        bestActionValue = -math.inf
        bestActionIndex = 0
        for index, value in enumerate(distribution):
            if index in self.simWorld.getPossibleActions():
                if value > bestActionValue:
                    bestActionValue = value
                    bestActionIndex = index
        return bestActionIndex

    def backPropogate(self, propogateValue: float):
        self.currentNode.totalEvaluation += propogateValue
        while self.currentNode.parent != None:
            self.currentNode = self.currentNode.parent

    def reRootTree(self):
        self.currentNode.parent = None


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

import math
import copy
from project2.MCTS.TreeNode import TreeNode
from project2.sim_world.sim_world import SimWorld


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
            if action not in self.simWorld.getPossibleActions():
                continue
            currentActionNodeValue = (-opponentFactor * self.currentNode.getExpectedResult(
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
                child=TreeNode(self.simWorld.getStateHash(),
                               self.simWorld.getMaxPossibleActionSpace())
            )
        
        playerTurn = self.simWorld.playerTurn
        self.makeSearchAction(self.treePolicyFindAction(playerTurn))

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
            playerTurn = self.simWorld.playerTurn
            treePolicyAction = self.treePolicyFindAction(playerTurn)
            currentNode = self.makeSearchAction(treePolicyAction)
        self.currentLeafNode = currentNode
        if not self.simWorld.isWinState():
            self.nodeExpansion()

        return currentNode

    def rollout(self, ANET):
        while not self.simWorld.isWinState():
            defaultPolicyAction = ANET.defaultPolicyFindAction(self.simWorld.getPossibleActions(), self.simWorld.getStateHash())
            self.simWorld.makeAction(defaultPolicyAction)
        return self.simWorld.getReward()

    

    def backPropogate(self, propogateValue: float):
        self.currentNode.totalEvaluation += propogateValue
        while self.currentNode.parent != None:
            #print("backPropogate",self.currentNode.parent)
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

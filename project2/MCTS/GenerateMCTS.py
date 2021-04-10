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
        self.ExplorationBiasCoefficient = 0
        self.HashTable = {}
        self.History = []

    def treePolicyFindAction(self) -> int:
        # TODO: Check if this works as expected
        # Or if we should use max() and min()
        bestAction = None
        bestValue = -math.inf
        for action in range(len(self.currentNode.numTakenAction)):
            if action not in self.simWorld.getPossibleActions():
                continue
            #print(self.currentNode.getExpectedResult(action),"bias", self.currentNode.getExplorationBias(action), "action",action, self.currentNode.state)
            #currentActionNodeValue = (opponentFactor * self.currentNode.getExpectedResult(
            #    action)) + self.ExplorationBiasCoefficient* self.currentNode.getExplorationBias(action)
            currentActionNodeValue = (self.simWorld.playerTurn * self.getExpectedResult(action)) + self.ExplorationBiasCoefficient* self.getExplorationBias(action)
            #print(currentActionNodeValue, self.currentNode.getExplorationBias(action), self.currentNode.numTakenAction[action])
            if(currentActionNodeValue > bestValue):
                bestValue = currentActionNodeValue
                bestAction = action
        return bestAction

    def nodeExpansion(self, action):
        self.currentNode.addChild(
            action=action,
            child=TreeNode(self.simWorld.peekAction(action),
                           self.simWorld.getMaxPossibleActionSpace())
        )

    def makeAction(self, action: int):
        self.simWorld.makeAction(action)
        self.currentNode.addActionTaken(action)       
        self.currentNode = self.currentNode.children.get(action)

        return self.currentNode

    def makeSearchAction(self, action: int):
        self.History.append([self.currentNode.state, action])
        self.currentNode.addActionTaken(action)
        self.currentNode = self.currentNode.children.get(action)
        self.simWorld.makeAction(action)

        self.currentNode.numTimesVisited += 1
        return self.currentNode

    def treeSearch(self, node: TreeNode, simWorld) -> TreeNode:
        self.simWorld = copy.deepcopy(simWorld)
        currentNode = self.currentNode
        currentNode.numTimesVisited += 1
        self.History = []
        nextAction = self.treePolicyFindAction()
        while nextAction in self.currentNode.children.keys():
            treePolicyAction = nextAction
            currentNode = self.makeSearchAction(treePolicyAction)
            nextAction = self.treePolicyFindAction()
        self.currentLeafNode = currentNode
        if not self.simWorld.isWinState():
            self.nodeExpansion(nextAction)
        return currentNode

    def rollout(self, ANET):
        defaultPolicyAction = 0

        while not self.simWorld.isWinState():
            defaultPolicyAction = ANET.defaultPolicyFindAction(self.simWorld.getPossibleActions(), self.simWorld.getStateHash())
            self.History.append([str(self.simWorld.getStateHash()), defaultPolicyAction])
            self.simWorld.makeAction(defaultPolicyAction)
        self.addToTable(self.simWorld.getStateHash(), self.simWorld.getReward(), defaultPolicyAction, self.simWorld.getMaxPossibleActionSpace())
        return self.simWorld.getReward()

    def backPropogate(self, propogateValue: float):
        for i in self.History:
            self.addToTable(i[0], propogateValue, i[1], self.simWorld.getMaxPossibleActionSpace())
        while self.currentNode.parent != None:
            self.currentNode.totalEvaluation += propogateValue
        
            #print("backPropogate",self.currentNode.parent)
            self.currentNode = self.currentNode.parent

    def reRootTree(self):
        self.currentNode.parent = None

    def addToTable(self, stateHash, reward, action, totalActions):
        if str(stateHash) in self.HashTable.keys():
            self.HashTable[str(stateHash)][0] = self.HashTable[str(stateHash)][0] + reward
            self.HashTable[str(stateHash)][1] += 1
            self.HashTable[str(stateHash)][2] [action] +=  1
        else:
            self.HashTable[str(stateHash)] = [reward, 1, [0] * totalActions]
            self.HashTable[str(stateHash)][2] [action] +=  1

        
    def getExpectedResult(self, action: int) -> float:
        #print(self.children[action].totalEvaluation, self.numTakenAction[action])
        peekAction = str(self.simWorld.peekAction(action))
        if peekAction not in self.HashTable.keys() or str(self.simWorld.getStateHash()) not in self.HashTable.keys():
            return 0
        return self.HashTable[peekAction][1] / (self.HashTable[str(self.simWorld.getStateHash())][2][action] + 1)
        
    def getExplorationBias(self, action: int) -> float:
        
        if str(self.simWorld.getStateHash()) not in self.HashTable.keys():
            return 1
        return  math.sqrt(math.log(self.HashTable[str(self.simWorld.getStateHash())][1]) / (self.HashTable[str(self.simWorld.getStateHash())][2][action] + 1))


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

import math
import copy
from typing import List

from project2.MCTS.TreeNode import TreeNode
from project2.sim_world.sim_world import SimWorld

import random

class MCTS:
    def __init__(
        self,
        root: TreeNode,
        ExplorationBias: float
    ):
       # self.simWorld = SimWorld(root.state)
        self.rootNode = root
        self.currentNode = root
        self.currentLeafNode = root
        self.ExplorationBiasCoefficient = ExplorationBias
        self.HashTable = {}
        self.History = []

    def treePolicyFindAction(self) -> int:
        bestActions = []
        bestValue = -math.inf
        for action in range(len(self.currentNode.numTakenAction)):
            if action not in self.simWorld.getPossibleActions():
                continue
            currentActionNodeValue = (self.simWorld.playerTurn * self.getExpectedResult(action)) + self.ExplorationBiasCoefficient* self.getExplorationBias(action)
            if currentActionNodeValue > bestValue:
                bestValue = currentActionNodeValue
                bestActions = [action]
            elif currentActionNodeValue == bestValue:
                bestActions.append(action)
        if len(bestActions) == 0:
            return None
        chosenAction = bestActions[random.randint(0, len(bestActions) -1)]
        return chosenAction

    def nodeExpansion(self, action):
        self.currentNode.addChild(
            action=action,
            child=TreeNode(self.simWorld.peekAction(action),
                           self.simWorld.getMaxPossibleActionSpace())
        )
        self.makeSearchAction(action)
    def makeAction(self, action: int):
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
        self.currentPlayer = self.simWorld.playerTurn
        
        self.currentNode.numTimesVisited += 1
        self.History = []
        nextAction = self.treePolicyFindAction()

        while nextAction in self.currentNode.children.keys():
            self.currentNode = self.makeSearchAction(nextAction)
            nextAction = self.treePolicyFindAction()
        self.currentLeafNode = self.currentNode
        if not self.simWorld.isWinState():
            self.nodeExpansion(nextAction)
        return self.currentNode

    def rollout(self, ANET):
        defaultPolicyAction = 0

        while not self.simWorld.isWinState():
            defaultPolicyAction = ANET.defaultPolicyFindAction(self.simWorld.getPossibleActions(), self.simWorld.getStateHash())
            self.History.append([str(self.simWorld.getStateHash()), defaultPolicyAction])
            self.simWorld.makeAction(defaultPolicyAction)
        #print(f"Rollout win. Reward: {self.simWorld.getReward()}. Player: won {self.simWorld.playerTurn * -1}")
        self.addToTable(self.simWorld.getStateHash(), self.simWorld.getReward(), defaultPolicyAction, self.simWorld.getMaxPossibleActionSpace())
        self.History.reverse()
        return self.simWorld.getReward()

    def backPropogate(self, propogateValue: float):
        for i in self.History:
            self.addToTable(i[0], propogateValue, i[1], self.simWorld.getMaxPossibleActionSpace())
        while self.currentNode.parent != None:
            self.currentNode.totalEvaluation += propogateValue
            self.currentNode = self.currentNode.parent

    def reRootTree(self):
        if self.currentNode != None:
            self.currentNode.parent = None

    def normaliseActionDistribution(self, stateHash) -> List:
        actionDistributtion = []
        actionSum =0
        for i in self.HashTable[stateHash][2]:
            actionDistributtion.append(i)
            actionSum += i

        # Normalise action distribution
        for i in range(len(actionDistributtion)):
            actionDistributtion[i] = (actionDistributtion[i]) / (actionSum)

        return actionDistributtion

    def addToTable(self, stateHash, reward, action, totalActions):
        if str(stateHash) in self.HashTable.keys():
            self.HashTable[str(stateHash)][0] = self.HashTable[str(stateHash)][0] + reward
            self.HashTable[str(stateHash)][1] += 1
            self.HashTable[str(stateHash)][2] [action] +=  1
        else:
            self.HashTable[str(stateHash)] = [reward, 1, [0] * totalActions]
            self.HashTable[str(stateHash)][2] [action] +=  1

    def getExpectedResult(self, action: int) -> float:
        if action not in self.simWorld.getPossibleActions():
            return self.simWorld.playerTurn * -math.inf
        peekAction = self.simWorld.peekAction(action)
        if peekAction == None:
            return -self.simWorld.playerTurn * math.inf
        if str(peekAction) not in self.HashTable.keys() or str(self.simWorld.getStateHash()) not in self.HashTable.keys():
            return 0
        return self.HashTable[str(peekAction)][0] / self.HashTable[str(peekAction)][1]
        
    def getExplorationBias(self, action: int) -> float:
        
        if str(self.simWorld.getStateHash()) not in self.HashTable.keys():
            return 1
        return  math.sqrt(math.log(self.HashTable[str(self.simWorld.getStateHash())][1]) / (self.HashTable[str(self.simWorld.getStateHash())][2][action] + 1))
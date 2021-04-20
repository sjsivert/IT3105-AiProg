import json
import math
import torch
from project2.sim_world.nim.Nim import Nim
from project2.MCTS.TreeNode import TreeNode
from project2.sim_world.sim_world import SimWorld
from project2.MCTS.GenerateMCTS import MCTS
from project2.Models.NeuralNet import NeuralActor
from project2.Models import SaveLoadModel
from project2.sim_world.hex.Hex import Hex
from typing import List, Tuple
from project2.Client_side.BasicClientActor import BasicClientActor
from project2.visualization.boardAnimator import BoardAnimator
import random
from typing import List
from project2.Models.SaveLoadModel import SaveModel, SaveTorchModel
import copy
import time

class ReinforcementLearningSystem:
    def __init__(
            self,
            numberOfTreeGames: int,
            numberOfGames: int,
            saveInterval:int,
            ANET:NeuralActor,
            explorationBias:float,
            RBUFsamples:int,
            exponentialDistributionFactor:float,
            simWorldTemplate: SimWorld,
            fileName: str,
            visualizeBoardWhileRunning: bool,
            visualizeInterval: int,
    ):
        self.RBuffer = []
        self.numberOfTreeGames = numberOfTreeGames
        self.numberOfGames = numberOfGames
        self.saveInterval = saveInterval
        self.ANET = ANET
        self.explorationBias = explorationBias
        self.RBUFsamples = RBUFsamples
        self.exponentialDistributionFactor = exponentialDistributionFactor
        self.simWorldTemplate = simWorldTemplate
        self.fileName = fileName
        self.visualizeBoardWhileRunning = visualizeBoardWhileRunning
        self.visualizeInterval = visualizeInterval

    def trainNeuralNet(self, numberOfGames, anetGenerationNumber):
        self.BoardVisualizer = BoardAnimator(self.simWorldTemplate.boardWidth)
        print("Training neuralnet")
        SaveTorchModel(self.ANET.neuralNet, self.fileName + str(anetGenerationNumber))
        for game in range(0 + anetGenerationNumber, numberOfGames + anetGenerationNumber):
            print(f"Playing game number: {game}")
            simWorld = copy.deepcopy(self.simWorldTemplate)
            # Random start player
            if(0.5> random.uniform(0,1)):
                simWorld.playerTurn = -1
            currentState = simWorld.getStateHash()
            root = TreeNode(state=currentState, parent=None, possibleActions = simWorld.getMaxPossibleActionSpace())
            mcts = MCTS(
                root=root,
                ExplorationBias = self.explorationBias
            )
            while not simWorld.isWinState():
                for e in range(self.numberOfTreeGames):
                    #startTime = time.time()
                    mcts.treeSearch(currentState, simWorld)
                    #endTime = time.time()
                    #print(f"Time ctrs.treeSearch {endTime - startTime}")
                    startTime = time.time()
                    reward = mcts.rollout(self.ANET)
                    endTime = time.time()
                    #print(f"Time rollout: {endTime - startTime}")
                    mcts.backPropogate(reward)

                actionDistribution =  mcts.normaliseActionDistribution(stateHash=str(simWorld.getStateHash()))

                self.RBuffer.append((mcts.currentNode.state, actionDistribution))

                bestMove = self.chooseActionPolicy(
                    actionDistribution=actionDistribution,
                    simWorld=simWorld,
                )
                #print("SW, BM", simWorld.state, (bestMove+1), actionDistribution)

                # Sync both sim worlds to be equal
                mcts.simWorld = copy.deepcopy(simWorld)
                mcts.makeAction(bestMove)
                simWorld.makeAction(bestMove)
                mcts.reRootTree()
                if (game) % self.visualizeInterval == 0 and self.visualizeBoardWhileRunning:
                    self.BoardVisualizer.addAnimationState(simWorld.getStateHash())
            if (game) % self.visualizeInterval == 0 and self.visualizeBoardWhileRunning:
                self.BoardVisualizer.animateEpisode()
                self.BoardVisualizer.clearEpisodes()
            self.ANET.trainOnRBUF(RBUF = self.RBuffer, minibatchSize = self.RBUFsamples, exponentialDistributionFactor = self.exponentialDistributionFactor)
            if (game) % self.saveInterval == 0 and game != 0:
                print(f"--------------SAVING MODEL: {self.fileName + str(game)}--------------")
                print(f"RBUF Size : {len(self.RBuffer)}")
                SaveTorchModel(self.ANET.neuralNet, self.fileName + str(game))
        self.playAgainstAnet()


    def playAgainstAnet(self):
        simWorld = copy.deepcopy(self.simWorldTemplate)
        simWorld.playAgainst(self.ANET)

    def chooseActionPolicy(self, actionDistribution, simWorld) -> int:
        bestMove = 0
        bestMove = None
        bestMoveValue = -math.inf
        
        #print(actionDistribution)
        for move in range(len(actionDistribution)):
            #print(actionDistribution[move], move)
            if bestMoveValue <actionDistribution[move] and move in simWorld.getPossibleActions():
                bestMoveValue = actionDistribution[move]
                bestMove = move
        return bestMove

    def saveModel(self):
        SaveTorchModel(self.ANET,"turnamentTrained")

    def mctsSearch(self, simWorld):
        if self.numberOfTreeGames > 1:
            state =simWorld.getStateHash()
            mcts = MCTS(
                root= TreeNode(
                    state = state,
                    parent = None,
                    possibleActions=simWorld.getMaxPossibleActionSpace(),
                ),
                ExplorationBias=1
            )
            for gameNr in range(self.numberOfTreeGames):
                mcts.treeSearch(state,simWorld)
                reward = mcts.rollout(self.ANET)
                mcts.backPropogate(reward)

            actionDistributtion = mcts.normaliseActionDistribution(stateHash=str(simWorld.getStateHash()))
            #print("Action dist", actionDistributtion)
        else:
            actionDistributtion = self.ANET.getDistributionForState(simWorld.getStateHash())[0]
        # TODO: Use different action policy for tournament?

        bestAction = self.chooseActionPolicy(
            actionDistribution=actionDistributtion,
            simWorld=simWorld
        )
        return bestAction, actionDistributtion

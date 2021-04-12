import json
import math
from project2.sim_world.nim.Nim import Nim
from project2.MCTS.TreeNode import TreeNode
from project2.sim_world.sim_world import SimWorld
from project2.MCTS.GenerateMCTS import MCTS
from project2.Models.NeuralNet import NeuralActor
from project2.Models.RandomAgent import RandomAgent
from project2.Models import SaveLoadModel
from project2.sim_world.hex.Hex import Hex
from project2.Tournament.LocalTournament import LocalTournament
from typing import List
from project2.Client_side.BasicClientActor import BasicClientActor
import random
from typing import List
from project2.Models.SaveLoadModel import SaveModel
import copy


RBUF = []
fileName = "test"

def main():
    # Load parameters from file
    with open('project2/parameters.json') as f:
        parameters = json.load(f)

    operationMode = parameters["operation_mode"]
    gameType = parameters['game_type']
    boardType = parameters['board_type']
    boardSize = parameters['board_size']

    learningRate = parameters['anet_learning_rate']
    activationFunction = parameters['anet_activation_function']
    outputActivationFunction = parameters['output_activation_function']
    optimizer = parameters['anet_optimizer']
    hiddenLayersDim = parameters['anet_hidden_layers_and_neurons_per_layer']
    lossFunction = parameters['loss_function']

    explorationBias = parameters['explorationBias']
    epsilon = parameters['epsilon']
    RBUFsamples = parameters['RBUFsamples']
    exponentialDistributionFactor = parameters['exponentialDistributionFactor']

    numEpisodes = parameters['mcts_num_episodes']
    numSearchGamesPerMove = parameters['mcts_n_of_search_games_per_move']
    saveInterval = parameters['save_interval']

    numCachedToppPreparations = parameters['anet_n_cached_topp_preparations']
    numToppGamesToPlay = parameters['anet_n_of_topp_games_to_be_played']

    if gameType == "hex":
        simWorld = Hex(
            boardType=boardType,
            boardWidth=boardSize,
            playerTurn=1,
            # loadedHexBoardState=[-1, 0, 0, 0, 1, -1, 0, 0, 0, 0],
        )
        input_size =  (boardSize * boardSize) + 1
        output_size = boardSize * boardSize

        #simWorld.playGame()

    elif gameType == "nim":
        simWorld = Nim(
            boardSize,
            2
        )
        input_size =  boardSize + 1
        output_size = 2
        #nim.playGayme()
    else:
        print("Game not specified. Quitting...")
    # is = save interval for ANET (the actor network) parameters
    if(operationMode == "play"):
        simWorld.playGame()

    elif (operationMode == "train"):

        print(input_size, output_size, hiddenLayersDim, learningRate)
        ANET = NeuralActor(
            input_size = input_size,
            output_size = output_size,
            hiddenLayersDim = hiddenLayersDim,
            learningRate = learningRate,
            lossFunction = lossFunction,
            optimizer = optimizer,
            activation = activationFunction,
            outputActivation = outputActivationFunction
        )
        doGames(
            numberOfTreeGames = numSearchGamesPerMove,
            numberOfGames = numEpisodes,
            saveInterval = saveInterval,
            ANET = ANET,
            explorationBias = explorationBias,
            epsilon = epsilon,
            RBUFsamples = RBUFsamples,
            exponentialDistributionFactor = exponentialDistributionFactor,
            simWorldTemplate = simWorld
)
    elif operationMode == "tournament":
        bsa = BasicClientActor(verbose=True)
        bsa.connect_to_server()
    else:
        raise Exception("Operation  mode not specified choose (play/train)")

def testTournament():
    agent1 = RandomAgent()
    agent2 = RandomAgent()
    simWorld = Nim(12, 2)
    testTournament = LocalTournament([agent1, agent2], numberOfFourGames = 5, roundRobin =  False, simWorldTemplate= simWorld, agentNames={agent1: "agent1", agent2: "agent2"})
    testTournament.runTournament()

def doGames(
        numberOfTreeGames: int,
        numberOfGames: int,
        saveInterval:int,
        ANET:NeuralActor,
        explorationBias:float,
        epsilon :float,
        RBUFsamples:int,
        exponentialDistributionFactor:float,
        simWorldTemplate: SimWorld) -> None:

    print(numberOfGames)
    for game in range(numberOfGames):
        RBUF = []
        print(game)
        simWorld = copy.deepcopy(simWorldTemplate)
        if(0.5> random.uniform(0,1)):
            simWorld.playerTurn = -1
        currentState = simWorld.getStateHash()
        root = TreeNode(state=currentState, parent=None, possibleActions = simWorld.getMaxPossibleActionSpace())
        mcts = MCTS(
            root=root,
            ExplorationBias = explorationBias
        )
        while not simWorld.isWinState():
            for e in range(numberOfTreeGames):
                mcts.treeSearch(currentState, simWorld)
                reward = mcts.rollout(ANET)
                mcts.backPropogate(reward)
            actionDistributtion = []
            actionSum =0
            for i in mcts.HashTable[str(simWorld.getStateHash())][2]:
                actionDistributtion.append(i)
                actionSum += i
            for i in range(len(actionDistributtion)):
                actionDistributtion[i] = (actionDistributtion[i]) / (actionSum)
            RBUF.append((mcts.currentNode.state, actionDistributtion))
            bestMove = 0
            if epsilon > random.uniform(0, 1) and (not simWorld.isWinState()):
                if len(simWorld.getPossibleActions()) > 1:
                    bestMove = simWorld.getPossibleActions()[random.randint(0, len(simWorld.getPossibleActions()) - 1)]
            else:
                bestMove = None
                bestMoveValue = -math.inf
                for move in range(len(actionDistributtion)):
                    if bestMoveValue < actionDistributtion[move] and move in simWorld.getPossibleActions():
                        bestMoveValue = actionDistributtion[move]
                        bestMove = move
            print("SW, BM", simWorld.state, (bestMove+1), actionDistributtion, simWorld.getPlayerTurn())
            mcts.simWorld = copy.deepcopy(simWorld)
            mcts.makeAction(bestMove)
            simWorld.makeAction(bestMove)
            mcts.reRootTree()
        ANET.trainOnRBUF(RBUF = RBUF, minibatchSize = RBUFsamples, exponentialDistributionFactor = exponentialDistributionFactor)
        if (game + 1) % saveInterval == 0:
            SaveModel(ANET.neuralNet, fileName + str(game))
    #TODO remove play against when done with code
    for i in range(1, 12 +1):
        nonstaones = [0] * (12 - i)
        nonnonstaones = [1] * (i)
        state = [-1] + nonstaones + nonnonstaones 
        state2 = [1] + nonstaones + nonnonstaones 
        print(i, ANET.getDistributionForState(state), ANET.defaultPolicyFindAction([0,1],state))
        print(i, ANET.getDistributionForState(state2), ANET.defaultPolicyFindAction([0,1],state2))
    simWorld2 = copy.deepcopy(simWorldTemplate)
    simWorld2.playAgainst(ANET)

if __name__ == '__main__':
    print("Run!")
    testTournament()
    #main()

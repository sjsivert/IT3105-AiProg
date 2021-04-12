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
from project2.RLS import ReinforcementLearningSystem


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
    fileNamePrefix = parameters['file_name']

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

    # Initiate Neural Net
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
    # Initiate ReinforcementLearningSystem
    RLS = ReinforcementLearningSystem(
            numberOfTreeGames = numSearchGamesPerMove,
            numberOfGames = numEpisodes,
            saveInterval = saveInterval,
            ANET = ANET,
            explorationBias = explorationBias,
            epsilon = epsilon,
            RBUFsamples = RBUFsamples,
            exponentialDistributionFactor = exponentialDistributionFactor,
            simWorldTemplate = simWorld,
            fileName = gameType + str(boardSize) + fileNamePrefix
    )
    # is = save interval for ANET (the actor network) parameters
    if(operationMode == "play"):
        print("Operation mode: Play")
        simWorld.playGame()

    elif (operationMode == "train"):
        print("Operation mode: train")
        print(input_size, output_size, hiddenLayersDim, learningRate)
        RLS.trainNeuralNet(numberOfGames=numEpisodes)
    elif operationMode == "tournament":
        print("Operation mode: Tournament")
        bsa = BasicClientActor(
            verbose=True,
            RLS = RLS
        )
        bsa.connect_to_server()
    else:
        raise Exception("Operation  mode not specified choose (play/train)")

def testTournament():
    agent1 = RandomAgent()
    agent2 = RandomAgent()
    agent3 = RandomAgent()
    simWorld = Nim(12, 2)
    testTournament = LocalTournament([agent1, agent2, agent3], numberOfFourGames = 5, roundRobin =  True, simWorldTemplate= simWorld, agentNames={agent1: "agent1", agent2: "agent2", agent3: "agent3"})
    testTournament.runTournament()


if __name__ == '__main__':
    print("Run!")
    main()

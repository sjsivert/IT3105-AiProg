import json
import math
import random
from sim_world.nim.Nim import Nim
from MCTS.TreeNode import TreeNode
from sim_world.sim_world import SimWorld
from MCTS.GenerateMCTS import MCTS
from Models.NeuralNet import NeuralActor
from Models import SaveLoadModel
from sim_world.hex.Hex import Hex
from typing import List
from Models.SaveLoadModel import SaveModel


RBUF = []
RBUFSamples = 10
fileName = "test"

def doGames(numberOfTreeGames: int, numberOfGames: int, saveInterval, input_size: int, output_size: int, hiddenLayersDimension: List, learningRate: int, simWorld: SimWorld) -> None:
    print(input_size, output_size, hiddenLayersDimension, learningRate)
    ANET = NeuralActor(input_size, output_size, hiddenLayersDimension, learningRate, 0.1)
    print(numberOfGames)
    for i in range(numberOfGames):
        RBUF = []
        ANET.epsilon = ANET.epsilon*0.9
        print(i)
        simWorld = Nim(
            20,
            2
        )
        currentState = simWorld.getStateHash()
        root = TreeNode(state=currentState, parent=None, possibleActions = output_size)
        mcts = MCTS(
            root=root
        )
        while not simWorld.isWinState():
            
            #monteCarloSimWorld = SimWorld(root)
            for e in range(numberOfTreeGames):
                mcts.treeSearch(currentState, simWorld)
                reward = mcts.rollout(ANET) * mcts.rootNode.state[0]
                #print("\n\n\n\n\n\n reward", reward, "\n\n\n\n\n\n")
                mcts.backPropogate(reward)
            actionDistributtion = mcts.currentNode.numTakenAction
            # print(RBUF)
            actionSum =0
            actionMin = 0
            for i in actionDistributtion:
                actionSum += i
                actionMin = min(actionMin, i)
            for i in range(len(actionDistributtion)):
                actionDistributtion[i] = (actionDistributtion[i] - actionMin) / (actionSum - actionMin)
            print(mcts.currentNode.state, actionDistributtion)
            # print(simWorld.playerTurn, simWorld.state)
            RBUF.append((mcts.currentNode.state, actionDistributtion))
            
            # TODO add epsilon
            if 0.0 > random.uniform(0, 1):
                if simWorld.state >= 2:
                    bestMove = random.randrange(0, len(actionDistributtion))
                else: bestMove = 0
            else:
                bestMove = None
                bestMoveValue = -math.inf
                #print("state", simWorld.state)
                for move in range(len(actionDistributtion)):
                    if bestMoveValue < actionDistributtion[move]:
                        bestMoveValue = actionDistributtion[move]
                        bestMove = move
            # print("SW, BM", simWorld.state, (bestMove+1), actionDistributtion)
            simWorld.makeAction(bestMove)
            mcts.makeAction(bestMove)
            mcts.reRootTree()

        ANET.trainOnRBUF(RBUF, minibatchSize = RBUFSamples)
        #print(RBUF)
        #if numberOfGames % saveInterval == 0:
            #weights = []
            #for nodeIndex, weight in enumerate(ANET.neuralNet.parameters()):
            #    weights.append(weight)
            #SaveModel(weights, fileName)

        
            # TODO Save ANET’s current parameters for later use in tournament play
    for i in range(0,21):
        print( ANET.defaultPolicyFindAction([1,2],[-1,i]))
        print( ANET.defaultPolicyFindAction([1,2],[ 1,i]))

    
    simWorld2 = Nim(
            20,
            2
        )
    simWorld2.playAgainst(ANET)
def main():
    # Load parameters from file
    with open('project2/parameters.json') as f:
        parameters = json.load(f)

    gameType = parameters['game_type']
    boardType = parameters['board_type']
    boardSize = parameters['board_size']
    boardSize = parameters['board_size']
    numEpisodes = parameters['mcts_num_episodes']
    numSearchGamesPerMove = parameters['mcts_n_of_search_games_per_move']
    saveInterval = parameters['save_interval']
    learningRate = parameters['anet_learning_rate']
    activationFunction = parameters['anet_activation_function']
    optimizer = parameters['anet_optimizer']
    hiddenLayersDim = parameters['anet_hidden_layers_and_neurons_per_layer']
    numCachedToppPreparations = parameters['anet_n_cached_topp_preparations']
    numToppGamesToPlay = parameters['anet_n_of_topp_games_to_be_played']
    if gameType == "hex":
        simWorld = Hex(
            boardType=boardType,
            boardWith=boardSize,
            playerTurn=1
            
        )
        input_size =  (boardSize * boardSize) + 1
        output_size = boardSize * boardSize

        #simWorld.playGame()

    elif gameType == "nim":
        simWorld = Nim(
            20,
            2
        )
        input_size =  2
        output_size = 2
        #nim.playGayme()
    else:
        print("Game not specified. Quitting...")
    # is = save interval for ANET (the actor network) parameters

    doGames(
        numberOfTreeGames = numSearchGamesPerMove,
        numberOfGames = numEpisodes, 
        saveInterval = saveInterval, 
        input_size =  input_size,
        output_size = output_size,
        hiddenLayersDimension= hiddenLayersDim,
        learningRate = learningRate,
        simWorld = simWorld
    )
    # clear replay buffer (RBUF)

    # randomly initialize parameters for ANET

    # for each number in actial games

    # simWorld = Initialize the actual game board to an empty board

    # currentState = startingBoardState (trengs denne?)

    # while simWorld not in final state
    # MTCS = initialize monte carlo sim world to same as root
    # for each number_search_games:
    # use three policy Pi to search from root to leaf
    # update MTCS.simWorld with each move

    # use ANET
    

if __name__ == '__main__':
    print("Run!")
    main()

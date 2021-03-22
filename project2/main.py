import json
import math
from sim_world.nim.Nim import Nim
from MCTS.TreeNode import TreeNode
from sim_world.sim_world import SimWorld
from MCTS.GenerateMCTS import MCTS
from Models.NeuralNet import NeuralActor
from Models import SaveLoadModel
from sim_world.hex.Hex import Hex


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
        simWorld.playGame()

    elif gameType == "nim":
        nim = Nim(
            10,
            3
        )
        nim.playGayme()
    else:
        print("Game not specified. Quitting...")
    # is = save interval for ANET (the actor network) parameters

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

RBUF = []
RBUFSamples = 10

fileName = "test"
inputSize = 10
layers = [2, 20, 3]
learningRate = 0.1



def doGames(self, rolloutsPerLeaf: int, numberOfTreeGames: int, numberOfGames: int, saveInterval) -> None:
    #TODO Initialize neural net
    ANET = NeuralActor(inputSize,layers, learningRate)

    for i in range(numberOfGames):
        simWorld = SimWorld()
        currentState = simWorld.__str__()
        root = TreeNode(state=currentState, parent=None)
        mcts = MCTS(
            root=root
        )
        while not simWorld.isWinState():
            #monteCarloSimWorld = SimWorld(root)
            for i in range(numberOfTreeGames):
                mcts.treeSearch(currentState, simWorld.playerTurn)
                reward = mcts.rollout()
                mcts.backPropogate(reward)
            actionDistributtion = mcts.currentNode.numTakenAction
            RBUF.append((mcts.simWorld.getStateHash(), actionDistributtion))

            #TODO add epsilon
            bestMove = None
            bestMoveValue = -math.inf
            for move in range(len(actionDistributtion)):
                if bestMoveValue < actionDistributtion[move]:
                    bestMoveValue = actionDistributtion[move]
                    bestMove = move

            mcts.makeAction(bestMove)
            mcts.reRootTree()

        #TODO Train ANET on a random minibatch of cases from RBUF
        ANET.trainOnRBUF(RBUF, RBUFSamples)
        if numberOfGames % saveInterval == 0:
            saveInterval.SaveModel(ANET.neuralNet.parameters, fileName)
            #TODO Save ANET’s current parameters for later use in tournament play

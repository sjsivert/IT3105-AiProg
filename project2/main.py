import json
from sim_world.nim.Nim import Nim
from TreeNode import TreeNode
from sim_world import SimWorld


def main():
    # Load parameters from file
    with open('parameters.json') as f:
        parameters = json.load(f)

    gameType = parameters['game_type']
    boardSize = parameters['board_size']
    numEpisodes = parameters['mcts_num_episodes']
    numSearchGamesPerMove = parameters['mcts_n_of_search_games_per_move']
    learningRate = parameters['anet_learning_rate']
    activationFunction = parameters['anet_activation_function']
    optimizer = parameters['anet_optimizer']
    hiddenLayersDim = parameters['anet_hidden_layers_and_neurons_per_layer']
    numCachedToppPreparations = parameters['anet_n_cached_topp_preparations']
    numToppGamesToPlay = parameters['anet_n_of_topp_games_to_be_played']

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
    # main()
    nim = Nim(
        10,
        3
    )
    nim.playGayme()


def doGames(self, rolloutsPerLeaf: int, numberOfTreeGames: int, numberOfGames: int) -> None:
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
                leafNode = mcts.treeSearch(currentState, simWorld.playerTurn)
                buffer = mcts.rollout()

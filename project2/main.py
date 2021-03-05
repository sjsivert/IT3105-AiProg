import json
from sim_world.nim.Nim import Nim


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

    # Train the model


if __name__ == '__main__':
    print("Run!")
    nim = Nim()
    main()

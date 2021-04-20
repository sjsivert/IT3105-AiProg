import math
from project2.visualization.boardAnimator import BoardAnimator
from project2.Models.SaveLoadModel import LoadTorchModel
from project2.Client_side.BasicClientActorAbs import BasicClientActorAbs
from project2.sim_world.hex.Hex import Hex
from project2.sim_world.nim.Nim import Nim
from project2.MCTS.TreeNode import TreeNode
from project2.sim_world.sim_world import SimWorld
from project2.MCTS.GenerateMCTS import MCTS
from project2.Models.NeuralNet import NeuralActor


class BasicClientActor(BasicClientActorAbs):

    def __init__(
            self,
            RLS,
            IP_address=None,
            verbose=True,
            visualizeBoardWhileRunning=False,
    ):
        self.RLS = RLS
        self.series_id = -1
        self.BoardVisualizer = BoardAnimator(boardSize=6)
        self.visualizeBoardWhileRunning = visualizeBoardWhileRunning
        BasicClientActorAbs.__init__(self, IP_address, verbose=verbose)
        self.RBUF = []
        self.RBUFSamples = 40
        self.lastState = None

    def handle_get_action(self, state):
        """
        Here you will use the neural net that you trained using MCTS to select a move for your actor on the current board.
        Remember to use the correct player_number for YOUR actor! The default action is to select a random empty cell
        on the board. This should be modified.
        :param state: The current board in the form (1 or 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), where
        1 or 2 indicates the number of the current player.  If you are player 2 in the current series, for example,
        then you will see a 2 here throughout the entire series, whereas player 1 will see a 1.
        :return: Your actor's selected action as a tuple (row, column)
        """
        # This is an example player who picks random moves. REMOVE THIS WHEN YOU ADD YOUR OWN CODE !!
       # next_move = tuple(self.pick_random_free_cell(
        #    state, size=int(math.sqrt(len(state)-1))))
        
        
        
        stateWithCorrectPlayers = tuple(
            map(lambda x: -1 if (x == 2) else (1 if (x == 1) else 0), state))

        playerTurn = 1 if stateWithCorrectPlayers[0] == 1 else -1
        simWorld = Hex(
            boardType="diamond",
            boardWidth=6,
            playerTurn=playerTurn,
            loadedHexBoardState=stateWithCorrectPlayers
        )
        if self.lastState == None:
            self.lastState = simWorld.getStateHash()
            for i in range(1, len(self.lastState)):
                if self.lastState[i] != 0:
                    self.lastState[i] = 0
                    break
        distributionOponent = [0] * simWorld.getMaxPossibleActionSpace()
        for key, value in simWorld.possibleActions.items():
            if simWorld.state.state[value[0]][value[1]].pegValue != 0:
                distributionOponent[key] = 1
        self.RBUF.append((self.lastState,distributionOponent))
        


        # Add animation frame
        self.BoardVisualizer.addAnimationState(simWorld.getStateHash())

        # Find best action
        actionNumber,distribution = self.RLS.mctsSearch(simWorld=simWorld)
        self.RBUF.append((simWorld.getStateHash(),distribution))

        # Convert local action nummer to expected coordinates
        coordinates = simWorld.getActionCoordinates(actionNumber)
        #print(f"SimWorld coordinates: {coordinates}")
        actionCordinatesConverted = simWorld.state.simWorldToTournament[coordinates]
        #print("Action coordinate chosen: ", actionCordinatesConverted)
        self.lastState = simWorld.peekAction(actionNumber)
        # self.BoardVisualizer.animateEpisode()

        return actionCordinatesConverted

    def handle_series_start(self, unique_id, series_id, player_map, num_games, game_params):
        """
        Set the player_number of our actor, so that we can tell our MCTS which actor we are.
        :param unique_id - integer identifier for the player within the whole tournament database
        :param series_id - (1 or 2) indicating which player this will be for the ENTIRE series
        :param player_map - a list of tuples: (unique-id series-id) for all players in a series
        :param num_games - number of games to be played in the series
        :param game_params - important game parameters.  For Hex = list with one item = board size (e.g. 5)
        :return
        """
        print(f"---------You are Player: {series_id}------------")
        self.series_id = series_id
        self.currentSeries = []
        #############################
        #
        #
        # YOUR CODE (if you have anything else) HERE
        #
        #
        ##############################

    def handle_game_start(self, start_player):
        """
        :param start_player: The starting player number (1 or 2) for this particular game.
        :return
        """
        self.starting_player = start_player
        #############################
        #
        #
        # YOUR CODE (if you have anything else) HERE
        #
        #
        ##############################

    def handle_game_over(self, winner, end_state):
        """
        Here you can decide how to handle what happens when a game finishes. The default action is to print the winner and
        the end state.
        :param winner: Winner ID (1 or 2)
        :param end_state: Final state of the board.
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        ##############################
        print("Game over, these are the stats:")
        print('Winner: ' + str(winner))
        print('End state: ' + str(end_state))
        # Animate
        self.RLS.ANET.trainOnRBUF(self.RBUF, self.RBUFSamples)
        if (self.visualizeBoardWhileRunning):
            self.BoardVisualizer.animateEpisode()
        self.BoardVisualizer.clearEpisodes()


    def handle_series_over(self, stats):
        """
        Here you can handle the series end in any way you want; the initial handling just prints the stats.
        :param stats: The actor statistics for a series = list of tuples [(unique_id, series_id, wins, losses)...]
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("Series ended, these are the stats:")
        print(str(stats))

    def handle_tournament_over(self, score):
        """
        Here you can decide to do something when a tournament ends. The default action is to print the received score.
        :param score: The actor score for the tournament
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        self.RLS.saveModel()
        print("Tournament over. Your score was: " + str(score))

    def handle_illegal_action(self, state, illegal_action):
        """
        Here you can handle what happens if you get an illegal action message. The default is to print the state and the
        illegal action.
        :param state: The state
        :param action: The illegal action
        :return:
        """
        ############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("An illegal action was attempted:")
        print('State: ' + str(state))
        print('Action: ' + str(illegal_action))


if __name__ == '__main__':
    bsa = BasicClientActor(verbose=True)
    bsa.connect_to_server()

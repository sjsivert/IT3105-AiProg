from typing import List, Dict
from project2.Tournament.TournamentPlotter import TournamentPlotter
import copy

class LocalTournament:

    def __init__(self, agents: List, numberOfFourGames:int, roundRobin:bool, simWorldTemplate, agentNames: Dict):
        self.agents = agents
        self.numberOfGames = numberOfFourGames
        self.roundRobin = roundRobin
        self.simWorldTemplate = simWorldTemplate
        self.agentNames = agentNames
        self.TournamentPlotter = TournamentPlotter(self.agentNames)

    def runTournament(self):
        print("Tournament start")
        if(len(self.agents) >= 2):
            if self.roundRobin:
                totalWins = {}
                for agent in self.agents:
                    totalWins[agent] = 0
                for i in range(len(self.agents)-1):
                    for j in range(i, len(self.agents)):
                        for numGames in range(self.numberOfGames):
                            results = self.playFourGames(self.agents[i], self.agents[j])
                            for agent in results.keys():
                                totalWins[agent] += results[agent]
                                print("Four game results: ", self.agentNames[self.agents[i]], ": ", results[self.agents[i]], "wins, ", 
                                self.agentNames[self.agents[j]], ": ", results[self.agents[j]], "wins.")
                self.printTotalWins(totalWins)
                self.TournamentPlotter.plottWins(totalWins)
            else:
                for i in range(len(self.agents)-1):
                    pairWins = {self.agents[i]: 0, self.agents[i+1]: 0}
                    for numGames in range(self.numberOfGames):
                            results = self.playFourGames(self.agents[i], self.agents[i+1])
                            for agent in results.keys():
                                pairWins[agent] += results[agent]
                                print("Four game results: ", self.agentNames[self.agents[i]], ": ", results[self.agents[i]], "wins, ", 
                                self.agentNames[self.agents[i+1]], ": ", results[self.agents[i+1]], "wins.")
                self.printTotalWins(pairWins)
                self.TournamentPlotter.plottWins(pairWins)
        print("Tournament over")

    def printTotalWins(self, totalWins):
        print("Tournament results")
        for agent, wins in totalWins.items():
            print(self.agentNames[agent], ": ", wins, " number of wins")

    def playFourGames(self, agent1, agent2):  # First and second as 1 and -1 = 4 games
        wins = {agent1: 0, agent2: 0}
        playerDict = {-1: agent1, 1: agent2}
        wins[self.playGame(playerDict, -1)] += 1
        wins[self.playGame(playerDict, 1)] += 1
        playerDict = {1: agent1, -1: agent2}
        wins[self.playGame(playerDict, -1)] += 1
        wins[self.playGame(playerDict, 1)] += 1
        return wins

    def playGame(self, playerDict: Dict, startingPlayer: int):
        simWorld = copy.deepcopy(self.simWorldTemplate)
        simWorld.playerTurn = startingPlayer
        while not simWorld.isWinState():  # Noen vinner altid? Mulig vi trenger en til sjekk. Random krasjer hvis possible = 0
            action = playerDict[simWorld.playerTurn].defaultPolicyFindAction(possibleActions = simWorld.getPossibleActions(), state = simWorld.getStateHash)
            simWorld.makeAction(action)
        return playerDict[-simWorld.playerTurn]
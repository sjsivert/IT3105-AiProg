import matplotlib.pyplot as plt

class TournamentPlotter:
    
    def __init__(self, agentNames):
        self.agentNames = agentNames
    
    def plottWins(self, wins):
        data = []
        names = []
        for agent, wins in wins.items():
            data.append(wins)
            names.append(self.agentNames[agent])
            plt.bar(names, data)
        plt.xlabel('Agent name')
        plt.ylabel('Nr of wins')
        plt.title('Tournament results')
        plt.show()
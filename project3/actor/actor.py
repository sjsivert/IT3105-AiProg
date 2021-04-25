import random

class Actor:

    def __init__(self, learningRate, discFactor, traceDecay, tileSize = [5, 5]):
        self.policy = []
        for i in range(5):  # Kjører 5 grids til å starte med.
            self.policy.append({})
        self.eligibility = {}
        self.learningRate = learningRate
        self.discFactor = discFactor
        self.traceDecay = traceDecay
        self.gridOffset = [[0,0], [0.4 , 0.2], [0.2, 0.6], [0.6, 0.8], [0.8, 0.4]]
        self.tileSize = tileSize
        self.actions = [-1, 0, 1]

    def getNextAction(self, state, epsilon):
        actionDistribution = [0, 0, 0] # action -1, 0, +1
        print("State: ", state)
        for gridNr in range(len(self.gridOffset)):
            #print(" --- ")
            tileKey = self.getTileKey(state, gridNr)
            #print("state:   ", state)
            #print("tileKey: ", tileKey)
            policy = self.getPolicy(gridNr, tileKey)
            print(gridNr, " policy: ", policy)
            for i in range(len(actionDistribution)):
                actionDistribution[i] += policy[i] * 1/len(self.gridOffset)
        print(actionDistribution, "\n----------- ")
        if random.random() < epsilon:   # Do random action, probable if high epsilon
            return random.randrange(0, 3)
        return actionDistribution.index(max(actionDistribution))
    
    def getPolicy(self, gridNr, tileKey):
        self.initPolicy(gridNr, tileKey)
        return self.policy[gridNr][tileKey]

    def getTileKey(self, state, gridNr):
        offset = [self.gridOffset[gridNr][0]*self.tileSize[0], self.gridOffset[gridNr][1]*self.tileSize[1]]
        return ((state[0]-offset[0])- ((state[0]-offset[0])%self.tileSize[0]) + offset[0], (state[1]-offset[1]) - ((state[1]-offset[1])%self.tileSize[1])+ offset[1])

    def updatePolicy(self, state, action, TDError):
        for gridNr in range(len(self.gridOffset)):
            tileKey = self.getTileKey(state, gridNr)
            self.initPolicy(gridNr, tileKey)
            self.policy[gridNr][tileKey][action] += self.learningRate * TDError #* self.eligibility[state]

    def initPolicy(self, gridNr, tileKey):
        if tileKey not in self.policy[gridNr]:
            self.policy[gridNr][tileKey] = [1, 1, 1]
    
    def reset(self):
        self.eligibility = {}

# Usikker på om det under funker.
    def decayEligibility(self, state, action):
        self.eligibility[state][action] = self.eligibility[state][action] * self.traceDecay * self.discFactor

    def setEligibility(self, state, action):
        if state not in self.eligibility:
            self.eligibility[state] = [0, 0, 0]
        self.eligibility[state][action] = 1

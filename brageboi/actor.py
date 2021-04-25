import random


class Actor:

    def __init__(self, learningRate, discFactor, traceDecay):
        self.policy = {}
        self.eTrace = {}
        self.learningRate = learningRate
        self.discFactor = discFactor
        self.traceDecay = traceDecay

    def reset(self):
        self.eTrace = {}

    def getNextAction(self, nextStates, epsilon):
        expectedValues = []
        for state in nextStates:
            self.setPolicy(state)
            expectedValues.append(self.policy[state])
        if random.random() < epsilon:   # Do random action, probable if high epsilon
            return nextStates[random.randrange(0, len(nextStates))]
        return nextStates[expectedValues.index(max(expectedValues))]

    def setETrace(self, state):
        if state not in self.eTrace:
            self.eTrace[state] = 1

    def updateETrace(self, state):
        self.eTrace[state] = self.eTrace[state] * self.traceDecay * self.discFactor

    def setPolicy(self, state):
        if state not in self.policy:
            self.policy[state] = 1

    def updatePolicy(self, state, TDError):
        self.policy[state] = self.policy[state] + self.learningRate * TDError * self.eTrace[state]

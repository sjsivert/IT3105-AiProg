import random


class TableCritic:

    def __init__(self, learningRate, discFactor, traceDecay):
        self.learningRate = learningRate
        self.discFactor = discFactor
        self.traceDecay = traceDecay
        self.expValues = {}
        self.eTrace = {}

    def resetETrace(self):
        self.eTrace = {}

    def setETrace(self, state):
        if state not in self.eTrace:
            self.eTrace[state] = 1

    def updateETrace(self, state):
        self.eTrace[state] = self.eTrace[state] * self.traceDecay * self.discFactor

    def setExpValues(self, state):
        if state not in self.expValues:
            self.expValues[state] = random.uniform(0.1, 0.2)

    def updateExpValues(self, state, TDError, isCurrentState=False):
        self.expValues[state] = self.expValues[state] + self.learningRate * TDError * self.eTrace[state]

    def getExpValues(self, state):
        return self.expValues[state]

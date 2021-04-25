import random
from criticTable import TableCritic
from criticNetwork import NetworkCritic
from parameters import Parameters as p


class Critic:

    def __init__(self, learningRate, discFactor, traceDecay):
        self.discFactor = discFactor
        self.traceDecay = traceDecay
        if p["useTableCritic"]:
            self.criticType = TableCritic(learningRate, discFactor, traceDecay)
        else:
            self.criticType = NetworkCritic(learningRate, discFactor, traceDecay, p["architecture"])

    def getTDError(self, reinforcement, state, stateTot):
        return reinforcement + self.discFactor * self.criticType.getExpValues(stateTot) \
               - self.criticType.getExpValues(state)  # r + yV(s') - V(s)

    def reset(self):
        self.criticType.resetETrace()

    def setETrace(self, state):
        self.criticType.setETrace(state)

    def updateETrace(self, state):
        self.criticType.updateETrace(state)

    def setExpValues(self, state):
        self.criticType.setExpValues(state)

    def updateExpValues(self, state, TDError, isCurrentState):
        self.criticType.updateExpValues(state, TDError, isCurrentState)

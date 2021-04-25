from parameters import Parameters as p, DiamondBoard as db, TriangleBoard as tb
import numpy as np
import torch


def stringToList(state):  # Turns string state into input layer
    out = []
    for s in state:
        out.append(int(s))
    return out


class NetworkCritic(torch.nn.Module):

    def __init__(self, learningRate, discFactor, traceDecay, architecture):
        super(NetworkCritic, self).__init__()
        self.weights = torch.nn.ModuleList()
        for num, val in enumerate(architecture):
            if num == 0:  # set of weights is between two layers
                continue
            self.weights.append(torch.nn.Linear(architecture[num - 1], val))
        self.learningRate = learningRate
        self.discFactor = discFactor
        self.traceDecay = traceDecay
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learningRate, momentum=0.0)

    def getExpValues(self, state):
        return self(torch.Tensor(stringToList(state)))

    def updateExpValues(self, state, TDError, isCurrentState=False):
        if isCurrentState:
            self.optimizer.zero_grad()
            outputs = self(torch.Tensor(stringToList(state)))  # Gradient given by current network (V(s))
            MSELoss = self.criterion(outputs + TDError, outputs)  # MSE given TDError and not
            MSELoss.backward(retain_graph=True)
            self.optimizer.step()

    def forward(self, inputLayer):
        layerNumber = 0
        for layer in self.weights:
            layerNumber += 1
            inputLayer = torch.nn.functional.relu(layer(inputLayer))  # __/
        return inputLayer

    def setETrace(self, state): pass

    def updateETrace(self, state): pass

    def setExpValues(self, state): pass

    def resetETrace(self): pass

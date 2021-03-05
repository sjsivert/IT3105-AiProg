from neuralnet.NeuralNet import NeuralNetwork
from neuralnet.NeuralNet import StateToArray
from sim_world.SimWorld import SAP

import torch
import torch.nn as nn
import math
import tensorflow as tf
import numpy as np
import torch.optim as optim


class CriticNeural:
    def __init__(self,
                 boardSize,
                 boardType,
                 hiddenLayersDim,
                 learningRate=0.1,
                 eligibilityDecay=0.9,
                 valueTable={},
                 discountFactor=0.9
                 ) -> None:
        self.valueTable = valueTable
        self.eligibilityDecay = eligibilityDecay
        self.eligibility = {}
        self.tdError = 0
        self.learningRate = learningRate
        self.discountFactor = discountFactor

        inputSize = 0
        if boardType == "triangle":
            inputSize = ((boardSize ** 2) + boardSize) / 2
        elif boardType == "diamond":
            inputSize = boardSize ** 2
        self.neuralNet = NeuralNetwork(
            input_size=inputSize,
            hiddenLayersDimension=hiddenLayersDim
        )
        # Optimizer stochastic gradient descent
        self.optimizer = optim.SGD(
            self.neuralNet.parameters(), lr=self.learningRate)

        self.lossFunction = nn.MSELoss()

    def resetEligibility(self):
        self.eligibility = {}

    def getValueTable(self) -> dict:
        return self.valueTable

    def getValue(self, state: list) -> float:
        state = StateToArray(state)
        return self.neuralNet(torch.tensor([int(s) for s in state], dtype=torch.float32)).item()

    # def setValue(self, key, value):
    #     self.valueTable[key] = value

    def updateTDError(self, reward, state, nextState):
        self.tdError = reward + \
            (self.discountFactor * self.getValue(nextState)) - self.getValue(state)

    def getEligibility(self, nodeIndex):
        if nodeIndex not in self.eligibility:
            self.eligibility[nodeIndex] = 0
        return self.eligibility[nodeIndex]

    def updateValue(self, stateActionPair: SAP):
        state = StateToArray(stateActionPair.state)

        # Map state to a pytorch friendly format
        input = torch.tensor(
            [int(s)for s in state], dtype=torch.float32)

        # We have to zero out gradients for each pass, or they will accumulate
        self.optimizer.zero_grad()
        output = self.neuralNet(input)

        # Avoid dividing ny zero
        if self.tdError == 0:
            self.tdError = 0.000000000001

        loss = self.lossFunction(output + self.tdError, output)

        # Store the gradients for the network
        loss.backward()

        # For each gradient in the network
        for nodeIndex, weight in enumerate(self.neuralNet.parameters()):
            # Get gradient for the weight and update it using eligibility
            self.eligibility[nodeIndex] = self.getEligibility(nodeIndex) + weight.grad * \
                ((-2 * 1/float(self.tdError)))
            weight.grad = float(self.tdError) * self.eligibility[nodeIndex]

        # Update the weights for the network using the gradients stored above
        self.optimizer.step()

    def decayEligibility(self, StateActionPair):
        for i in self.eligibility.keys():
            self.eligibility[i] = self.eligibility[i] * \
                self.discountFactor * self.eligibilityDecay

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import math
import random

netStructure = [{
    "type": "conv2d",
    "channels_in": 2,
    "channels_out": 16,
    "kernel": 3,
    "padding": 1,
    "activation": "relu"
},
{
    "type": "conv2d",
    "channels_in": 16,
    "channels_out": 32,
    "kernel": 3,
    "padding": 1,
    "activation": "relu"
},
{
    "type": "conv2d",
    "channels_in": 32,
    "channels_out": 64,
    "kernel": 3,
    "activation": "relu"
},
{
    "type": "flatten",
    "size": 256
},
{
    "type": "dense",
    "size": 128,
    "activation": "relu"
},
{
    "type": "dense",
    "size": 16,
    "activation": "softmax"
}]

class NeuralNetwork(nn.Module):

    def __init__(self,
            inputSize,
            outputSize,
            convLayersDim = [],
            denseLayersDim = []):
        self.convLayers = len(convLayersDim)
        self.totalLayers = len(convLayersDim) + len(denseLayersDim) + 1

        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for convLayer in convLayersDim:
            self.layers.append(nn.Conv2d(
            in_channels = convLayer["in"], 
            out_channels = convLayer["out"], 
            kernel_size = convLayer["kernel"], 
            padding = convLayer["padding"]))

        self.layers.append(nn.Flatten())
        denseInput = convLayersDim[-1]["out"] * (inputSize - 1)
        
        for numberOfNodes in denseLayersDim:
            self.layers.append(nn.Linear(in_features = denseInput, out_features= numberOfNodes))
            denseInput = numberOfNodes

        self.layers.append(nn.Linear(in_features = denseInput, out_features= outputSize))

    def forward(self, input):
        for index, layer in enumerate(self.layers):
            if index == self.totalLayers:
                input = F.softmax(layer(input))
            elif index == self.convLayers:
                input = layer(input)
            else:
                input = F.relu(layer(input))
        return input

class CCLoss(nn.Module):
    def init(self):
        super(CCLoss,self).init()

    def forward(self, x, y):
        return -(y * torch.log(x)).sum(dim=1).mean()
        
class NeuralActor:
    def __init__(self,
            input_size = None,
            output_size = None,
            convLayersDim = None,
            denseLayersDim = None,
            learningRate = None,
            lossFunction = None,
            optimizer = None,
            activation = None,
            outputActivation = None,
            model = None):
        if model == None:
            self.neuralNet = NeuralNetwork(
                inputSize= input_size,
                outputSize= output_size,
                convLayersDim= convLayersDim, 
                denseLayersDim = denseLayersDim)
        else:
            self.neuralNet = model

        if optimizer != None:
            if optimizer.lower() == "sgd":
                self.optimizer = optim.SGD(self.neuralNet.parameters(), lr = learningRate)
            elif optimizer.lower() == "adam":
                self.optimizer = optim.Adam(self.neuralNet.parameters(), lr = learningRate)
            elif optimizer.lower() == "rmsprop":
                self.optimizer = optim.RMSprop(self.neuralNet.parameters(), lr = learningRate)
            elif optimizer.lower() == "adagrad":
                self.optimizer = optim.Adagrad(self.neuralNet.parameters(), lr = learningRate)

        self.lossFunction = CCLoss()

    def trainOnRBUF(self, RBUF, minibatchSize:int, exponentialDistributionFactor = None): 
        minibatch = random.sample(RBUF, k=min(minibatchSize, len(RBUF)-1))

        for item in minibatch:
            state = self.structureInput(item[0])
            actionDistribution = item[1]
            input = torch.tensor(
                state, dtype=torch.float32)
            self.optimizer.zero_grad()
            output = self.neuralNet(input)
            loss = self.lossFunction(output, torch.tensor(actionDistribution))
            loss.backward(retain_graph = True)
            self.optimizer.step()

    def getDistributionForState(self, state: List):
        state = self.structureInput(state)
        input = torch.tensor(
            state, dtype=torch.float32)
        self.optimizer.zero_grad()
        output = self.neuralNet(input)
        # print("output", output)
        return output.detach().numpy()

    def defaultPolicyFindAction(self, possibleActions, state) -> int:
        distribution  = self.getDistributionForState(state)[0]
        return self.doStocasticChoice(distribution, possibleActions)
        '''#print("distrubution, state", distribution, state)
        bestActionValue = -math.inf
        bestActionIndex = 0
        for index, value in enumerate(distribution):
            if index in possibleActions:
                if value > bestActionValue:
                    bestActionValue = value 
                    bestActionIndex = index
        return bestActionIndex'''
    
    def structureInput(self, state):
        dim = int(math.sqrt(len(state)-1))
        player = []
        board = []
        for i in range(dim):
            player.append([])
            board.append([])
            for j in range(dim):
                player[i].append(state[0])
                board[i].append(state[1+i*dim+j])
        return([[player, board]])

    def doDeterministicChoice(self, distribution, possibleActions):
        bestActionValue = -math.inf
        bestActionIndex = 0
        for index in possibleActions:
            if distribution[index] > bestActionValue:
                bestActionValue = distribution[index] 
                bestActionIndex = index
        return bestActionIndex

    def doStocasticChoice(self, distribution, possibleActions):
        sum = 0
        for index in possibleActions:
            sum += distribution[index]
        rand = random.uniform(0, 1) * sum
        sum = 0
        for index in possibleActions:
            sum += distribution[index]
            if sum >= rand:
                return index
        return None

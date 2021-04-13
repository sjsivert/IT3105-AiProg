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
    "padding": 1,
    "activation": "relu"
},
{
    "type": "flatten",
    "size": 576
},
{
    "type": "dense",
    "size": 128,
    "activation": "relu"
},
{
    "type": "dense",
    "size": 9,
    "activation": "softmax"
}]

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        outputSize = 256
        for layer in netStructure:
            if(layer["type"] == "conv2d"):
                self.layers.append(nn.Conv2d(in_channels = layer["channels_in"], out_channels = layer["channels_out"], kernel_size = layer["kernel"], padding = layer["padding"]))
            elif(layer["type"] == "flatten"):
                outputSize = layer["size"]
                self.layers.append(nn.Flatten())  # Dobbelsjekk
            elif(layer["type"] == "dense"):
                self.layers.append(nn.Linear(in_features = outputSize, out_features= layer["size"]))  # Dobbelsjekk
                outputSize = layer["size"]

    def forward(self, input):
        # Hidden layers
        for index, layer in enumerate(self.layers):
            # print(layer)
            if netStructure[index]["type"] == "flatten":
                input = layer(input)
            elif netStructure[index]["activation"] == "relu":
                input = F.relu(layer(input))
            elif netStructure[index]["activation"] == "softmax":
                input = F.softmax(layer(input))
        #print(input)
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
            hiddenLayersDim = None,
            learningRate = None,
            lossFunction = None,
            optimizer = None,
            activation = None,
            outputActivation = None,
            model = None):
        if model == None:
            self.neuralNet = NeuralNetwork()
        else:
            self.neuralNet = model
        self.optimizer = optim.SGD(self.neuralNet.parameters(), lr = 0.01)
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

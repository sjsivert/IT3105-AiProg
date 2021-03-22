

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hiddenLayersDimension=[]):
        super(NeuralNetwork, self).__init__()

        self.layers = nn.ModuleList()

        current_dim = input_size

        for dimension in hiddenLayersDimension:
            print(int(current_dim), dimension)
            self.layers.append(nn.Linear(int(current_dim), dimension))
            current_dim = dimension

        self.layers.append(nn.Linear(current_dim, output_size))

    def forward(self, input):
        # Hidden layers
        for layer in self.layers[:-1]:
            input = F.relu(layer(input))

        # Output layer
        # Hyperbolic tangent
        out = torch.tanh(self.layers[-1](input))
        return 
        
class NeuralActor ():
    def __init__(self,
                 inputSize,
                 outputSize,
                 hiddenLayersDim,
                 learningRate=0.1
                 ):
        self.learningRate = learningRate

        self.neuralNet = NeuralNetwork(
            input_size=inputSize,
            hiddenLayersDimension=hiddenLayersDim,
            output_size = output_size
        )
        # Optimizer stochastic gradient descent
        self.optimizer = optim.SGD(
            self.neuralNet.parameters(), lr=self.learningRate)

        self.lossFunction = nn.MSELoss()


    def trainOnRBUF(self, RBUF, samples:int):
        #TODO choose random samples
        for i in range(len(RBUF)):
            state = RBUF[i][0]
            actionDistribution = RBUF[i][1]

            # Map state to a pytorch friendly format
            input = torch.tensor(
                [int(s)for s in state], dtype=torch.float32)

            # We have to zero out gradients for each pass, or they will accumulate
            self.optimizer.zero_grad()
            output = self.neuralNet(input)

            # Avoid dividing ny zero

            loss = self.lossFunction(actionDistribution, output)

            # Store the gradients for the network
            loss.backward()

            # Update the weights for the network using the gradients stored above
            self.optimizer.step()
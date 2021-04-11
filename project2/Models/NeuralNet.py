import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense

class NeuralActor ():
    
    def __init__(self,
                 inputSize,
                 outputSize,
                 hiddenLayersDim,
                 learningRate=0.1,
                 epsilon=0.0
                 ):
        self.learningRate = learningRate

        self.epsilon = epsilon

        self.neuralNet = self.getModel(
            input_size=inputSize,
            hiddenLayersDimension=hiddenLayersDim,
            output_size = outputSize
        )

    def getModel(self, input_size, output_size, hiddenLayersDimension=[]):
        model = Sequential()
        
        model.add(Dense(20, input_dim=input_size, kernel_initializer='he_uniform', activation='relu'))
        for i in range(len(hiddenLayersDimension)):
            model.add(Dense(hiddenLayersDimension[i]))
        model.add(Dense(output_size, activation='softmax'))
        model.compile(loss='mse', optimizer='adam')
        return model
    
    def trainOnRBUF(self, RBUF, minibatchSize:int):
        minibatch = random.sample(RBUF, k=min(minibatchSize, len(RBUF)-1))
        for item in minibatch:
            s = [[]]
            a = [[]]
            for i in item[0]:
                s[0].append(i)
            for i in item[1]:
                a[0].append(i)

            state = np.array(s)
            actionDistribution = np.array(a)
            self.neuralNet.fit(state, actionDistribution, verbose=0, epochs=100)

    def getDistributionForState(self, state: List):
        #print(state)
        #print(np.array(state))
        #print(np.array(state).transpose())
        s = [[]]
        for i in state:
            s[0].append(i)
        xList = np.array(s)
        yList = self.neuralNet.predict(xList)
        return(yList[0])


    def defaultPolicyFindAction(self, possibleActions, state) -> int:
        distribution  = self.getDistributionForState(state)
        #print(distribution, state)
        bestActionValue = -math.inf
        bestActionIndex = 0
        #print("dist",distribution)
        #print("possibel, state",possibleActions, state)
        for index, value in enumerate(distribution):
            #print("index, value", index, value)
            if index in possibleActions:
                if value > bestActionValue:
                    bestActionValue = value 
                    bestActionIndex = index
        #print("best", bestActionIndex, bestActionValue)
        return bestActionIndex
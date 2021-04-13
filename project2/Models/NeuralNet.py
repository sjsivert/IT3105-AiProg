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
from tensorflow import keras
from tensorflow.keras import layers

class NeuralActor ():
    
    def __init__(self,
                input_size = 0,
                output_size = 0,
                hiddenLayersDim = 0,
                learningRate:float = 0,
                lossFunction:str = "",
                optimizer:str = "",
                activation:str = "",
                outputActivation:str = "",
                model = None):
        if model == None:
            self.neuralNet = self.getModel(
                input_size=input_size,
                output_size = output_size,
                hiddenLayersDimension=hiddenLayersDim,
                learningRate = learningRate,
                lossFunction = lossFunction,
                optimizer = optimizer,
                activation = activation,
                outputActivation = outputActivation
            )
        else:
            self.neuralNet = model

    def getModel(self, 
                input_size, 
                output_size, 
                hiddenLayersDimension,
                learningRate:float,
                lossFunction:str,
                optimizer:str,
                activation:str,
                outputActivation:str):

        model = Sequential()
        
        model.add(Dense(20, input_dim=input_size, kernel_initializer='he_uniform', activation=activation.lower()))
        for i in range(len(hiddenLayersDimension)):
            model.add(Dense(hiddenLayersDimension[i]))
        model.add(Dense(output_size, activation=outputActivation.lower()))

        op = None
        if optimizer.lower() == "adam":
            op = keras.optimizers.Adam(learning_rate=learningRate)

        elif optimizer.lower() == "sgd":
            op = keras.optimizers.SGD(learning_rate=learningRate)
        
        model.compile(loss=lossFunction, optimizer =op)
        return model
    
    def trainOnRBUF(self, RBUF, minibatchSize:int, exponentialDistributionFactor:float):
        '''
        minibatch = []
        indices = list(range(0,len(RBUF)))
        for i in range(0, min(minibatchSize, len(RBUF)-1)):
            rand1 = random.uniform(0, 1)
            rand2 = random.uniform(0, 1)
            randomNumber = rand1 * (rand2 ** exponentialDistributionFactor) 
            sample = int(round(randomNumber * (len(indices) - 1)))
            minibatch.append(RBUF[indices[sample]])
            del indices[sample]'''
        #minibatch = random.sample(RBUF, k=min(minibatchSize, len(RBUF)-1))
        # Train on the last 50 games
        if (len(RBUF) > minibatchSize):
            minibatch = random.sample(RBUF[-minibatchSize:], k=min(minibatchSize, len(RBUF) - 1))
        else:
            minibatch = random.sample(RBUF, k=min(minibatchSize, len(RBUF) - 1))
        for item in minibatch:
            s = [[]]
            a = [[]]
            for i in item[0]:
                s[0].append(i)
            for i in item[1]:
                a[0].append(i)

            state = np.array(s)
            actionDistribution = np.array(a)
            self.neuralNet.fit(state, actionDistribution, verbose=0, epochs=200)

    def getDistributionForState(self, state: List):
        #print(state)
        #print(np.array(state))
        #print(np.array(state).transpose())
        s = [[]]
        for i in state:
            s[0].append(i)
        xList = np.array(s)
        yList = self.neuralNet(xList)
        return(yList[0])

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
    
    def defaultPolicyFindAction(self, possibleActions, state) -> int:
        distribution  = self.getDistributionForState(state)

        bestActionIndex = self.doDeterministicChoice(distribution, possibleActions)
        #bestActionIndex = self.doStocasticChoice(distribution, possibleActions)
        return bestActionIndex
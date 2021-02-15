
from GenerateBoard import BoardState
from tensorflow import keras
import tensorflow.keras.models as keras_model
import tensorflow.keras.layers as keras_layers
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers_dim=[]):
        super(NeuralNetwork, self).__init__()

        self.layers = nn.ModuleList()

        current_dim = input_size

        for hdim in hidden_layers_dim:
            print(int(current_dim), hdim)
            self.layers.append(nn.Linear(int(current_dim), hdim))
            current_dim = hdim

        self.layers.append(nn.Linear(current_dim, 1))

    def forward(self, tensor):
        # Input layer
        t = tensor

        # Hidden layers
        for layer in self.layers[:-1]:
            t = F.relu(layer(t))

        # Output layer
        out = torch.tanh(self.layers[-1](t))
        return out


"""
def generateNetwork(
    num_classes=7,  # what type of data we get as input
    lrate=0.01,
    opt='SGD',  # Optimation function for stochastic gradient descent
    loss='categorical_crossentropy',
    act='relu',  # Activation function
    conv=True,  # Convolution net or not
    lastact='softmax',  # last activation function for output,
    inShape=(7, )
):
    opt = eval('keras.optimizers.' + opt)
    #loss = eval('keras.losses' + loss) if type(loss) == str else loss
    input = keras.layers.Input(shape=inShape, name='inputLayer')
    x = input

    x = keras.layers.Dense(50, activation=act)(x)
#    x = keras.layers.Dense(25, activation=act)(x)
    output = keras.layers.Dense(num_classes, activation=lastact)(x)
    model = keras.models.Model(input, output)
    model.compile(
        optimizer=opt(lr=lrate),
        loss=loss
        # metrics=[keras.metrics.categorical_crossentropy] # TODO change
    )
    return model


def neuralNet(
    epochs=100,
    ncases=500,
    segl=8,
    vlen=50,
    vf=0.2,
    learningRate=0.001,
    act='relu',
    convolution=False,
    mbs=16,
    verb=1,
    tb=False
):
    # Data preparation
    # Length of target vectors
    neuralNet = generateNetwork()  # TODO: Pass inn parameters

    splitGD = SplitGD(neuralNet)  # Make the split-gradient-descent object
    splitGD.fit(
        inputs,
        targets,
        epochs=epochs,
        mbs=mbs,
        vfrac=vf,
        verbosity=verb,
        callbacks=cbs
    )


"""


def StateToArray(state: List) -> List[int]:
    pegList = []

    if state == None:
        return pegList

    for layer in state:
        for peg in layer:
            pegList.append(peg.pegValue)
    return pegList

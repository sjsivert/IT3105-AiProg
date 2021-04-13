import torch
from tensorflow import keras
from project2.Models.NeuralNet import NeuralActor
from project2.Models.NeuralNetDom import NeuralActor as NAD

def SaveModel(model, filename):
    model.save(filename)
    print("saved model", filename)


def LoadModel(fileName):
    model = keras.models.load_model(fileName)
    print("loaded model", fileName)
    return NeuralActor(model = model)

def LoadTorchModel(fileName):
    model = torch.load(fileName)
    print("loaded model", fileName)
    return NAD(model=model)
    
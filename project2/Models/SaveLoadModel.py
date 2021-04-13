import torch
import json
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
    with open('project2/parameters.json') as f:
        parameters = json.load(f)
    modelSaveLocation = parameters["model_save_location"]
    model = torch.load(modelSaveLocation+fileName)
    print("loaded model", fileName)
    return NAD(model=model)

def SaveTorchModel(model, fileName):
    with open('project2/parameters.json') as f:
        parameters = json.load(f)
    modelSaveLocation = parameters["model_save_location"]
    torch.save(model, modelSaveLocation+fileName)
    print("Saved model", fileName)

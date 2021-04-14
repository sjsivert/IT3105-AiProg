import torch
import json
from tensorflow import keras
from project2.Models.NeuralNet import NeuralActor
from project2.Models.NeuralNetDom import NeuralActor as NAD
from shutil import copyfile

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
    optimizer = parameters["anet_optimizer"]
    learningRate = parameters["anet_learning_rate"]
    lossFunction = parameters["loss_function"]

    model = torch.load(modelSaveLocation+fileName)

    print("loaded model", fileName)
    return NAD(
        model=model,
        optimizer=optimizer,
        learningRate=learningRate,
        lossFunction=lossFunction,
    )

def SaveTorchModel(model, fileName):
    with open('project2/parameters.json') as f:
        parameters = json.load(f)
    modelSaveLocation = parameters["model_save_location"]
    path = modelSaveLocation+fileName
    torch.save(model, path)
    copyParameterFile(path + "_parameters.json")
    print("Saved model", fileName)


def copyParameterFile(path: str) -> None:
    copyfile("./project2/parameters.json", path)

from tensorflow import keras
from project2.Models.NeuralNet import NeuralActor

def SaveModel(model, filename):
    model.save(filename)
    print("saved model", filename)


def LoadModel(fileName):
    model = keras.models.load_model(fileName)
    print("loaded model", fileName)
    return NeuralActor(model = model)
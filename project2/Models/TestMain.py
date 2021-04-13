from NeuralNet import NeuralActor
from NeuralNetDom import NeuralActor as NeuralActorDom
import time

def main():
    NN = NeuralActor(20, 2, [30,30,30, 10], 0.001,"mse", "sgd", "relu", "softmax")
    NN2 = NeuralActor(20, 2, [30,30,30, 10], 0.001,"mae", "adam", "relu", "softmax")
    RBUF = []

    for i in range (30):
        '''
        RBUF.append(([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], [1.0, 0.0]))
        RBUF.append(([0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], [0.0, 1.0]))
        RBUF.append(([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1], [0.5, 0.5]))
        RBUF.append(([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1], [1.0, 0.0]))

    #NN2.trainOnRBUF(RBUF, 20, 0)
    #NN2.trainOnRBUF(RBUF, 20, 0)
    #NN2.trainOnRBUF(RBUF, 20, 0)
    #NN2.trainOnRBUF(RBUF, 20, 0)
    NN2.trainOnRBUF(RBUF, 20, 0)
    NN2.trainOnRBUF(RBUF, 20, 0)
    NN2.trainOnRBUF(RBUF, 20, 0)

    print(RBUF[0][1], NN2.getDistributionForState(RBUF[0][0]))
    print(RBUF[1][1], NN2.getDistributionForState(RBUF[1][0]))
    print(RBUF[2][1], NN2.getDistributionForState(RBUF[2][0]))
    print(RBUF[3][1], NN2.getDistributionForState(RBUF[3][0]))'''
    
    NN = NeuralActor(10, 9, [30,30,30, 10], 0.001,"mse", "sgd", "relu", "softmax")
    print(NN.getDistributionForState([-1,-1,0,1,0,-1,0,0,-1,0]))
    print(NN.getDistributionForState([-1,-1,1,1,1,-1,0,1,-1,0]))
    RBUF = []
    for i in range(300):
        RBUF.append([[-1,-1,0,1,0,-1,0,0,-1,0],[0,0,  1,0,0,1,  1,1,  0]])
        RBUF.append([[-1,-1,1,1,1,-1,0,1,-1,0],[1,0,0.2,0,0.7,0.5,0,0,1]])
    NN.trainOnRBUF(RBUF,20, 0)
    print("Top", NN.getDistributionForState([-1,-1,0,1,0,-1,0,0,-1,0]))
    print("Bot", NN.getDistributionForState([-1,-1,1,1,1,-1,0,1,-1,0]))
    for i in range(500):
        NN.getDistributionForState([-1,-1,0,1,0,-1,0,0,-1,0])
    print("Done")
    

def main2():
    NN = NeuralActorDom()
    print(NN.getDistributionForState([-1,-1,0,1,0,-1,0,0,-1,0]))
    print(NN.getDistributionForState([-1,-1,1,1,1,-1,0,1,-1,0]))
    RBUF = []
    for i in range(300):
        RBUF.append([[-1,-1,0,1,0,-1,0,0,-1,0],[0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.25, 0.25, 0.0]])
    for epoch in range(50):
        NN.trainOnRBUF(RBUF,20)
    
    for i in range(300):
    
        RBUF.append([[-1,-1,1,1,1,-1,0,1,-1,0],[0.29411764705882354, 0.0, 0.05882352941176471, 0.0, 0.20588235294117646, 0.14705882352941177, 0.0, 0.0, 0.29411764705882354]])
    for epoch in range(50):
        NN.trainOnRBUF(RBUF,20)

    print("Top2", NN.getDistributionForState([-1,-1,0,1,0,-1,0,0,-1,0]))
    print("Bot", NN.getDistributionForState([-1,-1,1,1,1,-1,0,1,-1,0]))
    for i in range(500):
        NN.getDistributionForState([-1,-1,0,1,0,-1,0,0,-1,0])
    print("Done")
    



if __name__ == '__main__':
    print("Test!")
    deltaTime = time.time()
    main()
    print("Keras: ", time.time()- deltaTime)
    deltaTime = time.time()
    main2()
    print("Pytorch", time.time()- deltaTime)
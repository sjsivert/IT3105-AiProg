from NeuralNet import NeuralActor

def main():
    NN = NeuralActor(20, 2, [30,30,30, 10], 0.001,"mse", "sgd", "relu", "tanh")
    NN2 = NeuralActor(20, 2, [30,30,30, 10], 0.001,"mae", "adam", "relu", "tanh")
    RBUF = []

    for i in range (30):
        RBUF.append(([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], [1.0, 0.0]))
        RBUF.append(([0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], [0.0, 1.0]))
        RBUF.append(([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1], [0.5, 0.5]))
        RBUF.append(([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1], [1.0, 0.0]))
    '''
    NN.trainOnRBUF(RBUF, 20, 0)
    NN.trainOnRBUF(RBUF, 20, 0)
    NN.trainOnRBUF(RBUF, 20, 0)
    NN.trainOnRBUF(RBUF, 20, 0)
    NN.trainOnRBUF(RBUF, 20, 0)
    NN.trainOnRBUF(RBUF, 20, 0)
    NN.trainOnRBUF(RBUF, 20, 0)

    print(RBUF[0][1], NN.getDistributionForState(RBUF[0][0]))
    print(RBUF[1][1], NN.getDistributionForState(RBUF[1][0]))
    print(RBUF[2][1], NN.getDistributionForState(RBUF[2][0]))
    print(RBUF[3][1], NN.getDistributionForState(RBUF[3][0]))
    print()'''

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
    print(RBUF[3][1], NN2.getDistributionForState(RBUF[3][0]))

if __name__ == '__main__':
    print("Test!")
    main()
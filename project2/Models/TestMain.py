from NeuralNet import NeuralActor

def main():
    NN = NeuralActor(2, 2, [30, 10], 1)
    RBUF = [([1, 20], [1.0, 0.0])] * 20
    print(NN.getDistributionForState([1,20]))
    NN.trainOnRBUF(RBUF, 1)
    print("Boi", NN.getDistributionForState([1,20]))


if __name__ == '__main__':
    print("Test!")
    main()

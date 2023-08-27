from Network import Network, Layer_Dense, Activation_Function, ReLU, Softmax
import numpy as np
import copy
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd


def one_hot(Y):
    one_hot_Y = np.zeros(Y.size)
    one_hot_Y[np.where(Y == Y.max())[0]] = 1

    return one_hot_Y



def get_accuracy(output,Y):

    predictions = [np.argmax(i) for i in output]
    return np.sum(predictions == Y) / Y.size

if __name__ == "__main__":
    test = pd.read_csv('mnist_test.csv.zip')
    train = pd.read_csv('mnist_train.csv.zip')

    test = np.array(test)
    train = np.array(train)

    Y_train = train[:,0]
    X_train = train[:,1:]
    X_train = X_train/255

    net = Network()
    lay1 = Layer_Dense(784,10,0)
    lay1.weights = np.random.rand(784,10) - 0.5
    lay1.biases = np.random.rand(1,10) - 0.5

    lay2 = Layer_Dense(10,10,0)
    lay2.weights = np.random.rand(10,10) - 0.5
    lay2.biases = np.random.rand(1,10) - 0.5
    
    
    net.add_layer(lay1)
    net.add_layer(ReLU())
    net.add_layer(lay2)
    net.add_layer(Softmax())

    net.forward(X_train)

    print(get_accuracy(net.output,Y_train))

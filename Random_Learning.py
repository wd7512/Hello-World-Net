from Network import Network, Layer_Dense
import numpy as np
import copy
import matplotlib.pyplot as plt
from tabulate import tabulate


def Random_Mutate(Network,step = 0.01):

    for layer in Network.layers:
        if layer.__class__.__name__ == 'Layer_Dense':
            n_inputs,n_neurons = layer.weights.shape
            layer.weights += step*np.random.randn(n_inputs,n_neurons)

            n_inputs,n_neurons = layer.biases.shape
            layer.biases += step*np.random.randn(n_inputs,n_neurons)

    

def Random_Learn(net,gens,error_function,step):
    '''
    learn to add/multiply 2 numbers together
    '''

    learn_X = [(x,y) for x in range(100) for y in range(100)]

    error = []
    base_error = error_function(net,learn_X)
    for i in range(gens):
        #print(net)
        
        

        
        error.append(base_error)

        stuck = True
        while stuck:
            changed_net = copy.deepcopy(net)
            Random_Mutate(changed_net,step)

            changed_error = error_function(changed_net,learn_X)

            if changed_error < base_error:
                base_error = changed_error

                net = changed_net


                stuck = False
            

        print('Gen',i,' Error',changed_error)

    return net,error

def Addition_Error(net,learn_X):
    error = 0
    for x,y in learn_X:
        X = [x,y]

        net.forward(X)

        error += float(abs((x+y) - net.output[0]))

    return error

def Addition_Test():
    net = Network()
    net.add_layer(Layer_Dense(2,1))

    test_X = ([0,0],[0,1],[1,0],[1,1],[1000,0],[0,1000],[1000,1000])
    display = []
    for x in test_X:
        net.forward(x)
        display.append([x,net.output[0],abs((x[0]+x[1])-net.output[0])])

    print(net)
    net,err = Random_Learn(net,50,Addition_Error,0.1)

    print('Learning Complete')
    print(net)
    display = []
    for x in test_X:
        net.forward(x)
        display.append([x,float(net.output[0]),abs((x[0]+x[1])-net.output[0])])

    print(tabulate(display,headers=['Input','Output','Error']))

    return net

def Multiplication_Error(net,learn_X):
    error = 0
    for x,y in learn_X:
        X = [x,y]

        net.forward(X)

        error += float(abs((x*y) - net.output[0]))

    return error

def Multiplication_Test():
    net = Network()
    net.add_layer(Layer_Dense(2,2))
    net.add_layer(Layer_Dense(2,1))
    

    test_X = ([0,0],[0,1],[1,0],[1,1],[1000,0],[0,1000],[1000,1000])
    display = []
    for x in test_X:
        net.forward(x)
        display.append([x,net.output[0],abs((x[0]*x[1])-net.output[0])])

    print(net)
    net,err = Random_Learn(net,100,Multiplication_Error,0.2)

    print('Learning Complete')
    print(net)
    display = []
    for x in test_X:
        net.forward(x)
        display.append([x,float(net.output[0]),abs((x[0]*x[1])-net.output[0])])

    print(tabulate(display,headers=['Input','Output','Error']))

    return net

if __name__ == '__main__':
    net = Addition_Test()

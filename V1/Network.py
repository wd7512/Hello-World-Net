import numpy as np


class Network:
    def __init__(self):
        #initialise list to hold layers
        self.layers = []
        self.error_log = []
    def add_layer(self,layer):
        '''
        adds a layer to the network after existing layers
        '''
        self.layers.append(layer)

        # test if we are adding a valid layer
        test = self.check_integrity()

        if test == False:
            print('Invalid Layer Added')
            print(self.error_log[-1])
            raise Exception
        
    def forward(self,X):
        '''
        push X through the neural network
        '''

        if self.layers == []:
            return X
        else:
            for layer in self.layers:
                layer.forward(X)
                X = layer.output

        self.output = X

    def check_integrity(self):
        #checks integrity of the network
        
        if self.layers == []:
            #print('Network Passed Integrity Test')
            return True
        
        n_inputs,_ = self.layers[0].weights.shape

        X = np.random.randn(1,n_inputs)

        
        try:
            
            self.forward(X)
            #print('TEST: inputs = '+str(X)+' outputs = '+str(self.output))
        except Exception as e:
            #print('Network Failed Integrity Test')
            self.error_log.append(e)
            return False
        

        #print('Network Passed Integrity Test')
        return True

    def backwards_update(self,X,Y,step):
        self.forward(X)

        m = len(Y)

        dZ = self.output - Y

        K = len(self.layers)
        for i in range(len(layers)):
            j = K - i
            lay = self.layers[j]
            if lay.__class__.__name__ == 'layer_dense':
                #dW = np.dot(
        

    def __str__(self):
        display = ''
        for layer in self.layers:
            
            display = display + '\n---------- \n' +  layer.__str__() 

        return display + '\n---------- \n'
    
                



class Layer_Dense:
    def __init__(self,n_inputs,n_neurons,scale = 0.1):
        #initialise weights and biases
        self.weights = scale * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self,X):
        #push X through the function
        self.output = np.dot(X, self.weights) + self.biases

    def __str__(self):
        return self.__class__.__name__ + '\n' +'Weights: '+'\n'+str(self.weights)+'\n'+'Biases: '+'\n'+str(self.biases)



                

class Activation_Function:
    def __init__(self,function):
        

        if callable(function):
            #tests if it is a callable function
            self.function = function
        else:
            print('Invalid Function')
            raise Exception
        
    def forward(self,X):

        self.output = self.function(X)

    def __str__(self):

        return self.__class__.__name__ +'\n'+self.function.__name__

class ReLU(Activation_Function):
    def __init__(self):
        self.function = lambda x : np.maximum(0,x)

        
class Softmax(Activation_Function):
    def __init__(self):
        self.function = lambda x : np.exp(x) / np.sum(np.exp(x))

import numpy as np
import pickle

class network():
    '''
    class to hold a neural network
    '''
    def __init__(self,n_in,n_out):
        self.layers = [] #list to contain all layers
        self.n_in = n_in
        self.n_out = n_out
        self.mutateable_layers = [] #list to index mutateable layers
        
    def add_layer(self,layer): #adds a layer
        self.layers.append(layer)
        temp = self.check_integrity()
        if temp == False:
            print('Removing Added Layer')
            del self.layers[-1]
        else:
            if type(layer).__name__ == 'layer_dense' or type(layer).__name__ == 'layer_one_to_one':
                self.mutateable_layers.append(len(self.layers)-1)

        # need to check layer matches last layer
        
    def remove_layer(self,index): # removes layer at index
        del self.layers[index]

    def check_integrity(self): #check integrity of layers
        if self.layers == []:
            return True
        else:
            X = np.random.randn(1,self.n_in)
            try:
                self.forward(X)
                
            except Exception as e:
                print('Network Failed Integrity Test')
                print(e)
                return False
            return True
            
    def forward(self,X):
        '''
        Inputs must always be a 2d array of a batch of input vectors
        You can always use a singular input vector but must be contained in another array so it is 2d
        '''
        if np.shape(X)[1] != self.n_in:
            raise Exception('Wrong input size')
        else:
            self.output = X
            for layer in self.layers:
                self.output = layer.forward(self.output)
            return self.output

    def reset(self): #set all layers to 0
        for index in self.mutateable_layers:
            self.layers[index].__init__(self.layers[index].n_in,self.layers[index].n_out)

    def save_to_file(self, filename): 
        '''
        saves neural network
        remember to add .pkl to the filename
        '''
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def __str__(self): #print of the layers
        display = ''
        for layer in self.layers:
            
            display = display + '\n---------- \n' +  layer.__str__() 

        return display + '\n---------- \n'

class layer_dense(): #a dense neural layer 
    def __init__(self,n_in,n_out):
        # initialise weights and biases to 0
        self.biases = np.zeros(n_out)
        self.weights = np.zeros((n_in,n_out))
        self.n_in = n_in
        self.n_out = n_out
    
    def forward(self,X):
        # push values through the layer
        self.output = np.dot(X,self.weights) + self.biases
        return self.output

    
    def __str__(self): #prints info regarding the layer
        return self.__class__.__name__ + '\n' +'Weights: '+'\n'+str(self.weights)+'\n'+'Biases: '+'\n'+str(self.biases)
    
class layer_one_to_one(): # a one to one layer
    def __init__(self,n_in_out):
        # initialise weights and biases to 0
        self.n_in = n_in_out
        self.n_out = n_in_out
        self.biases = np.zeros(n_in_out)
        self.weights = np.zeros(n_in_out)

    def forward(self,X):
        # push values through the layer
        self.output = np.multiply(X,self.weights) + self.biases
        return self.output
    
class layer_dropout(): # a dropout layer
    def __init__(self,n_in_out,prob):
        self.n_in = n_in_out
        self.n_out = n_in_out
        self.weights = np.random.choice([0, 1], size=n_in_out, p=[1 - prob, prob])
        self.prob = prob
    
    def forward(self,X):
        # push values through the layer
        self.weights = np.random.choice([0, 1], size=self.n_in_out, p=[1 - self.prob, self.prob]) #create random dropout layer each time
        self.output = np.multiply(X,self.weights)
        return self.output


class activation_function():
    def __init__(self,function):
        if callable(function):
            self.function = function
        else:
            raise Exception('Input is not a function')
    
    def forward(self,X):
        self.output = np.apply_along_axis(self.function, 1, X)
        return self.output
    
    def __str__(self):
        return self.__class__.__name__ +'\n'+self.function.__name__

class relu(activation_function):
    def __init__(self):
        pass
    def function(self,x):
        return np.maximum(0,x)
        
class softmax(activation_function):
    def __init__(self):
        pass
    def function(self,x):
        return np.exp(x) / np.sum(np.exp(x))

class sigmoid(activation_function):
    def __init__(self):
        pass
    def function(self,x):
        return 1 / (1+np.exp(-x))


def load_network_from_file(filename):
    with open(filename, 'rb') as file:
        loaded_network = pickle.load(file)
    return loaded_network


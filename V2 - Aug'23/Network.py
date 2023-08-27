import numpy as np

class network():
    '''
    Inputs must always be a 2d array of a batch of input vectors
    You can always use a singular input vector but must be contained in another array so it is 2d
    '''
    def __init__(self,n_in,n_out):
        self.layers = [] #list to contain all layers
        self.n_in = n_in
        self.n_out = n_out
        self.mutateable_layers = []
        
    def add_layer(self,layer): #adds a layer
        self.layers.append(layer)
        temp = self.check_integrity()
        if temp == False:
            print('Removing Added Layer')
            del self.layers[-1]
        else:
            if type(layer).__name__ == 'layer_dense':
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
        if np.shape(X)[1] != self.n_in:
            raise Exception('Wrong input size')
        else:
            self.output = X
            for layer in self.layers:
                self.output = layer.forward(self.output)
            return self.output

    def __str__(self): #print of the layers
        display = ''
        for layer in self.layers:
            
            display = display + '\n---------- \n' +  layer.__str__() 

        return display + '\n---------- \n'

class layer_dense():
    def __init__(self,n_in,n_out):
        # initialise weights and biases to 0
        
        self.biases = np.zeros(n_out)
        self.weights = np.zeros((n_in,n_out))
    
    def forward(self,X):
        # push values through the layer
        self.output = np.dot(X,self.weights) + self.biases
        return self.output
    
    def __str__(self): #prints info regarding the layer
        return self.__class__.__name__ + '\n' +'Weights: '+'\n'+str(self.weights)+'\n'+'Biases: '+'\n'+str(self.biases)

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
        self.function = lambda x : np.maximum(0,x)
        
class softmax(activation_function):
    def __init__(self):
        self.function = lambda x : np.exp(x) / np.sum(np.exp(x))

# remove everything below this later

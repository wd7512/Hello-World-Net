import numpy as np
from copy import deepcopy
import plotly.express as px


class Layer_Dense: #hidden layer
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        #minus the largest value from each output to stop large exp values
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Brain:
    def __init__(self):
        self.lay1 = Layer_Dense(16,8)
        self.act1 = Activation_ReLU()
        self.lay2 = Layer_Dense(8,2)
        self.act2 = Activation_Softmax()
    def forward(self, inputs):
       
        self.lay1.forward(inputs)
        self.act1.forward(self.lay1.output)
        
        self.lay2.forward(self.act1.output)
        self.act2.forward(self.lay2.output)

        self.output = self.act2.output

def multi_mutate(brains, x): #keeps the brains and makes x many children asexually
    out = brains

    for i in range(len(brains)):
        brain = brains[i]
        for i in range(x):
            new_brain = deepcopy(brain)

            new_brain.lay1.weights += np.random.randn(16, 8) * np.random.randn(16, 8)
            new_brain.lay1.biases += 0.01 * np.random.randn(1,8)
            new_brain.lay2.weights += np.random.randn(8, 2) * np.random.randn(8, 2)
            new_brain.lay2.biases += 0.01 * np.random.randn(1,2)
        
            out.append(new_brain)

    return out

def learn(networks,gens):

    top_brains = int(networks/10) #top 10% stay alive in each gen
    trial_brains = [Brain() for i in range(networks)]
    output = []

    for i in range(gens):
        scores = []
        
        for brain in trial_brains: #THIS IS WHERE YOUR SCORING / LOSS FUNCTION GOES
            score = 'function' #score each brain
            scores.append(score)

        print('Gen :',i)
        print('Best Score :',max(scores))
        print('Avg Score :',sum(scores) / len(scores))

        max_index = scores.index(max(scores))
        best_brain = trial_brains[max_index]

        output.append([scores,best_brain])

        scores_copy = scores.copy()
        multi_best_brains = [] #list for best brains
        for x in range(top_brains): #picks out top 10% brains
            multi_best_brains.append(trial_brains[scores_copy.index(max(scores_copy))])

            scores_copy.remove(max(scores_copy))


        trial_brains = multi_mutate(multi_best_brains, 9) #mutate

    return output


def save_brain(brain,filename):
    out = np.array([brain.lay1.weights,
                    brain.lay1.biases,
                    brain.lay2.weights,
                    brain.lay2.biases
                    ])

    np.save(filename,out,)

def open_brain(filename):
    out = np.load(filename,allow_pickle=True)

    brain = Brain()
    brain.lay1.weights = out[0]
    brain.lay1.biases = out[1]
    brain.lay2.weights = out[2]
    brain.lay2.biases = out[3]

    return brain


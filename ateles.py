# The first Neural Net Model of the Ateles project
# Named after the spider monkeys subjected to efferent nerve swapping operations
# by Dr. Roger Wolcott Sperry

import numpy as np
import math
import copy

class DenseSequential:
    def __init__(self, n, activation):
        # The array n defines the width of each layer in the neural net
        # starting at the input nodes and ending with the final output
        self.width = n
        self.depth = len(n)-1
        # randomly initialize our weights with mean 0
        self.gen_synapses()
        self.f = getattr(self, activation)
    
    def gen_synapses(self):
        self.synapse = [];
        for i in range(0,self.depth):
            self.synapse.append(2*np.random.random((self.width[i],self.width[i+1])) - 1) 
    
    def sigmoid(self, x,deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))
    
    def softsign(self, x, deriv=False):
        if(deriv==True):
            return 1/(1+abs(x))**2
        return x/(1+abs(x))
        
    def swish(self,x,deriv=False):
        if(deriv==True):
            exp(self.beta*x)*(self.beta*x + exp(self.beta*x) + 1)/(exp(self.beta*x) + 1)**2
        return x/(1 + exp(-self.beta*x))

    def learn(self, X, y):
        # reduce error (single step) for the matrix inputs X and output y
        self.apply(X)
        E = self.layer[-1] - y
        self.error = E
        for i in range(self.depth, 0, -1):
            dfdw = self.f(self.layer[i],deriv=True)
            dE = E*dfdw
            # I'm aiming to make this look like Data Mining 6.4
            w_j = self.synapse[i-1].T
            E = dE.dot(w_j)
            a_j = self.layer[i-1] # The vector at the input to the weights_i,j
            self.synapse[i-1] -= a_j.T.dot(dE)
            
	        
    def boost(self, gain):
		# Based on the neural nets of stroke survivers, and children learning to walk
		# Increase the excitation of all the synapses when outputs are small
        for i in range(self.depth):
            self.synapse[i] = self.synapse[i]*gain
	
    def apply(self, X):
        self.layer = []
        self.layer.append(X)
        for synapse in self.synapse:
            self.layer.append(self.f(np.dot(self.layer[-1],synapse)))
        self.decision = self.layer[-1]
    
    def success_rate(self,X,y):
        self.apply(X)
        correct = 0
        samples = X.shape[0]
        for i in range(samples):
            if np.all(np.round(self.layer[-1][i]) == y[i]):
                correct += 1
        self.success_probability = correct/samples
        
    def grow(self,percent):
        for i in range(1,len(self.width)-1):
            self.width[i] = math.floor(self.width[i]*(1 + percent/100))
        saveSynapse = copy.deepcopy(self.synapse)
        self.gen_synapses()
        for i in range(len(saveSynapse)):
            x = saveSynapse[i].shape[0]
            y = saveSynapse[i].shape[1]
            self.synapse[i][0:x, 0:y] = saveSynapse[i][:,:]
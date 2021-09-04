import random
from math import *

class neuron():
    def __init__(self,n_neuron,bias,lr,weight_range):
        self.weight = []
        self.bias = bias
        self.error = 0
        self.output = 0.0
        self.inputs = []
        self.lr = lr
        
        for i in range(0,n_neuron):
            r = random.random() * weight_range
            self.weight.append(r)
    
    def getWeight(self):
        return self.weight
   
    def activate(self,inputs):
        self.output = self.bias
        self.inputs = inputs
        i = 0
        for inp in inputs:
                self.output += inp * self.weight[i]
                i = i + 1
        self.output = self.sigmoid(self.output)
        return self.output
    
    def sigmoid(self,output):
        return 1.0/(1.0 + exp(-output))
   
    def getError(self):
        return self.error
 
    def setError(self,error):
        self.error = error
    
    def update_weight(self):
        for i in range(0,len(self.weight)):
            self.weight[i] = self.weight[i] + self.lr * self.error * self.inputs[i]
            if self.weight[i] < 0:
                self.weight[i] = 0
    
    def output(self):
        return self.output
        
    
        

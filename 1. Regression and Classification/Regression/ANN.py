import random

class neuron():
    def __init__(self,n_neuron,bias,lr,n_weight):
        self.weight = []
        self.bias = bias
        self.error = 0
        self.output = 0.0
        self.inputs = []
        self.lr = lr
        
        for i in range(0,n_neuron):
            w = random.random() * n_weight 
            self.weight.append(w)
    
    def getWeight(self):
        return self.weight
    
    def activate(self,inputs):
        self.output = self.bias
        self.inputs = inputs
        a = 0
        for i in inputs:
                self.output += i * self.weight[a]
                a = a +1
        return self.output
    
    def getError(self):
        return self.error
    
    def setError(self,error):
        self.error = error
    
    def update_weight(self):
        for i in range(0,len(self.weight)): 
            self.weight[i] = self.weight[i] - self.lr * self.error * self.inputs[i]
            if self.weight[i] < 0:
                self.weight[i] = 0
    
    def output(self):
        return self.output
        
    
        

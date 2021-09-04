import random
from math import *
import ANN

class MLP():
    def __init__(self,n_input,n_hidden,n_nodes,lr):
        self.n_hidden = n_hidden
        self.n_nodes = n_nodes
        self.lr = lr
        self.cross_en = []
        
        weight_bias = 1
            
        self.layers = []
        for i in range(0,self.n_hidden):
            layer = [] 
            
            for j in range(0,self.n_nodes):
                n_perceptron = self.n_nodes
                bias = random.random()
                
                if i == 0:
                    n_perceptron = n_input
                node = ANN.neuron(n_perceptron,bias,self.lr,weight_bias)
                layer.append(node)
            self.layers.append(layer)

        output_layer = []

        output_layer.append(ANN.neuron(n_perceptron,0.0,self.lr,weight_bias))
        output_layer.append(ANN.neuron(n_perceptron,0.0,self.lr,weight_bias))
        self.layers.append(output_layer)
    
    def lossFunction(self):
        return self.cross_en
    
    def neuron_derivative(self,output):
        return output*(1.0-output)

    def forward_propagate(self,input_neuron):
        output_neuron = []
        for i in range(0,len(self.layers)):
            new_data = []
            for j in range(0,len(self.layers[i])):
                new_input = self.layers[i][j].activate(input_neuron)
                new_data .append(new_input)
            input_neuron = new_data
            output_neuron.append(new_data)

        return output_neuron
    
    def back_propagate(self,target_val,output):
        for i in reversed(range(0,len(self.layers))):
            errors = []
            if i != len(self.layers) - 1: 
                for j in range(0,len(self.layers[i])):
                    error = 0.0
                    for k in range(0,len(self.layers[i+1])):
                        e_weight = self.layers[i+1][k].getWeight()
                        e_error = self.layers[i+1][k].getError()
                        for w in range(0,len(e_weight)):
                            error += e_weight[w] * e_error
                    errors.append(error)
            else:
                for j in range(0,len(self.layers[i])):
                    cross_error = (target_val[j] - output[j])
                    errors.append(cross_error)
                
            for j in range(0,len(self.layers[i])):
                self.layers[i][j].setError(errors[j] * self.neuron_derivative(self.layers[i][j].output))

    
    def max_index(self,value):
        max_i = 0
        max_n = 0
        for i in range (0,len(value)):
            if value[i] >= max_n:
                max_n = value[i]
                max_i = i
        return max_i
    
    def valid(self,output,target_label):
        if output[0] == target_label[0] and output[1] ==target_label[1]:
            return 1
        else:
            return 0

    def training(self,train_data,epoch,target_data):
        
        for i in range(0,epoch):
            result_prediction = []
            prob_prediction = []
            sum_error = 0.0
            accuracy = 0
            for j in range(0,len(train_data)):
                output_neuron = self.forward_propagate(train_data[j])
                
                sum_log = 0.0
                prob_prediction= []
                for m in range(0,len(output_neuron[-1])):
                    out_m = output_neuron[-1][m]
                    prob_prediction.append(out_m)
                    sum_log += target_data[j][m]*log(max(prob_prediction[m],0.1))
                    
                sum_error += sum_log
                max_i = self.max_index(prob_prediction)

                prob_prediction[max_i] = 1
                for p in range(0,len(prob_prediction)):
                    if prob_prediction[p] != 1:
                        prob_prediction[p] = 0
                result_prediction.append(prob_prediction)
                accuracy += self.valid(prob_prediction,target_data[j])
                self.back_propagate(target_data[j],prob_prediction) 
                for k in range(0,len(self.layers)):
                    for l in range(0,len(self.layers[k])):
                        self.layers[k][l].update_weight()

            self.cross_en.append(-sum_error/(1.0*len(target_data)))
            accuracy = accuracy/(1.0*len(target_data))
            print( "Epoch-" + str(i) + ", Cross-entropy error : " + str(self.cross_en[-1]) + ", Accuracy " + str(accuracy))
               
        return result_prediction

    def testing(self,test_data,target_data):
        testing_result = []
        sum_error = 0.0
        accuracy = 0.0
        for i in range(0,len(test_data)):
            output_neuron = self.forward_propagate(test_data[i])
            predicted = []
            sum_log = 0.0
            for m in range(0,len(output_neuron[-1])):
                    out_m = output_neuron[-1][m]
                    predicted.append(out_m)
                    sum_log += target_data[i][m]*log(max(predicted[m],0.1))
            max_id = self.max_index(predicted)
            predicted[max_id] = 1
            for p in range(0,len(predicted)):
                if predicted[p] != 1:
                    predicted[p] = 0
            testing_result.append(predicted)
            accuracy += self.valid(predicted,target_data[i])
            sum_error += sum_log
        print ("Cross-entropy error : " + str(-sum_error/(1.0*len(target_data))) + "\n")
        print ("Accuracy : " + str(accuracy/(1.0*len(target_data))) + "\n")
        return testing_result


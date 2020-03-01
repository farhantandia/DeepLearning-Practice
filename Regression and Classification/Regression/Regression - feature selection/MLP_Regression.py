import random
from math import *
import ANN


class MLP():
    def __init__(self, n_input, n_hidden, n_nodes, lr):
        self.n_hidden = n_hidden
        self.n_nodes = n_nodes
        self.lr = lr
        self.rms = []
        self.cross_en = []
        weight_bias = 1e-8

        self.layers = []
        for i in range(0, self.n_hidden):
            layer = []

            for j in range(0, self.n_nodes):
                n_perceptron = self.n_nodes
                bias = random.random()

                if i == 0:
                    n_perceptron = n_input
                node = ANN.neuron(n_perceptron, bias, self.lr, weight_bias)
                layer.append(node)
            self.layers.append(layer)

    def lossFunction(self):
        return self.rms

    def forward_propagate(self, input_neuron):
        output_neuron = []
        for i in range(0, len(self.layers)):
            new_data = []
            for j in range(0, len(self.layers[i])):
                new_input = self.layers[i][j].activate(input_neuron)
                new_data.append(new_input)
            input_neuron = new_data
            output_neuron.append(new_data)
        return output_neuron

    def back_propagate(self, target_val, output):
        for i in reversed(range(0, len(self.layers))):

            errors = []
            if i != len(self.layers) - 1:
                error = 0.0
                for node in self.layers[i + 1]:
                    e_weight = node.getWeight()
                    e_error = node.getError()
                    for w in range(0, len(e_weight)):
                        error += e_weight[w] * e_error
                    errors.append(error)
            else:
                for j in range(0, len(self.layers[i])):
                    errors.append(self.layers[i][j].output - target_val)
            for j in range(0, len(self.layers[i])):
                if len(errors) < len(self.layers[i]):
                    self.layers[i][j].setError(errors[0])
                else:
                    self.layers[i][j].setError(errors[j])

    def training(self, train_data, epoch, target_data):

        for i in range(0, epoch):
            result_prediction = []
            prob_prediction = []
            sum_error = 0.0
            for j in range(0, len(train_data)):
                output_neuron = self.forward_propagate(train_data[j])

                predicted = output_neuron[-1][0]
                result_prediction.append(predicted)
                sum_error += pow((predicted - target_data[j]), 2)
                self.back_propagate(target_data[j], prob_prediction)
                for k in range(0, len(self.layers)):
                    for l in range(0, len(self.layers[k])):
                        self.layers[k][l].update_weight()

            self.rms.append(sqrt(sum_error / (1.0 * len(target_data))))
            print("Epoch-" + str(i) + ", Training RMSE : " + str(self.rms[-1]))
        return result_prediction

    def testing(self, test_data, target_data):
        testing_result = []
        sum_error = 0.0
        for i in range(0, len(test_data)):
            output_neuron = self.forward_propagate(test_data[i])
            predicted = output_neuron[-1][0]
            testing_result.append(predicted)
            sum_error += (predicted - target_data[i]) ** 2
        print("Testing RMSE : " + str(sqrt(sum_error / (1.0 * len(target_data)))) + '\n')
        return testing_result

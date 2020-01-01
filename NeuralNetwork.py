# Inspired by: https://towardsdatascience.com/artificial-neural-networks-optimization-using-genetic-algorithm-with-python-1fe8ed17733e
from settings_constants import activation
import numpy

def sigmoid(inpt):
    return 1.0 / (1.0 + numpy.exp(-1 * inpt))

def relu(inpt):
    return max(inpt,0)

def tanh(inpt):
    return numpy.tanh(inpt)

class NeuralNetwork:
    def __init__(self, inp = None, hidden = None, outp = None, weights_matrices = None):
        """ inp:        number of input neurons
            type:       int
            
            outp:       number of output neurons
            type:        int
                        
            hidden:     list with every element representing one hidden layer with the number of neurons given by the element itself
            type:       list of ints

            weights_matrices:   already predetermined weight matrices, makes other parameters irrelevant
            type:               list of numpy matrices
        """
        if weights_matrices is not None:
            self.weights_matrices = weights_matrices
        else:
            if not hidden:
                self.weights_matrices = [numpy.random.uniform(low=-1, high=1,size=(outp, inp+1))]
            else:
                self.weights_matrices = [numpy.random.uniform(low=-1, high=1,size=(hidden[x+1], hidden[x]+1)) for x in range(len(hidden)-1)]
                self.weights_matrices.insert(0, numpy.random.uniform(low=-1, high=1,size=(hidden[0], inp+1)))
                self.weights_matrices.append(numpy.random.uniform(low=-1, high=1,size=(outp, hidden[-1]+1)))


    def predict_outputs(self, data_inputs, activation_fct=activation):
        # activation function
        function = tanh
        if activation_fct == "relu":
            function = relu
        elif activation_fct == "sigmoid":
            function = sigmoid

        # appending bias, converting to numpy array
        previous_neurons_bias = numpy.append(numpy.array(data_inputs), 1)

        for idx, weight_matrix in enumerate(self.weights_matrices):
            # apply weights to calculate values for next layer
            next_neurons = numpy.matmul(weight_matrix, previous_neurons_bias)
            # activation with activation function
            if idx != len(self.weights_matrices)-1:
                next_neurons = numpy.array(list(map(function, next_neurons)))
            # appending bias
            previous_neurons_bias = numpy.append(next_neurons, 1)

        return previous_neurons_bias[0], previous_neurons_bias[1]
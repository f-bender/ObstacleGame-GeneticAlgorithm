# Inspired by: https://towardsdatascience.com/artificial-neural-networks-optimization-using-genetic-algorithm-with-python-1fe8ed17733e

import numpy

def sigmoid(inpt):
    return 1.0 / (1.0 + numpy.exp(-1 * inpt))

def relu(inpt):
    # result = inpt
    # result[inpt < 0] = 0
    return inpt if inpt > 0 else 0

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


        # self.inp_hl_weights = numpy.random.uniform(low=-0.1, high=0.1,size=(hidden, inp+1))
        # self.hl_outp_weights = numpy.random.uniform(low=-0.1, high=0.1,size=(outp, hidden+1)) # +1 for bias
        # self.all_weights = numpy.array([inp_hl_weights, hl_outp_weights])

    def predict_outputs(self, data_inputs, activation="sigmoid"):
        # activation function
        function = sigmoid if activation == "sigmoid" else relu

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



        # predictions = numpy.zeros(shape=(data_inputs.shape[0]))
        # print(predictions)
        # for sample_idx in range(data_inputs.shape[0]):
        #     r1 = data_inputs[sample_idx, :]
        #     for curr_weights in weights_mat:
        #         r1 = numpy.matmul(a=r1, b=curr_weights)
        #         if activation == "relu":
        #             r1 = relu(r1)
        #         elif activation == "sigmoid":
        #             r1 = sigmoid(r1)
        #     predicted_label = numpy.where(r1 == numpy.max(r1))[0][0]
        #     predictions[sample_idx] = predicted_label
        # correct_predictions = numpy.where(predictions == data_outputs)[0].size
        # accuracy = (correct_predictions / data_outputs.size) * 100
        # return accuracy, predictions

    def fitness(weights_mat, data_inputs, data_outputs, activation="relu"):
        accuracy = numpy.empty(shape=(weights_mat.shape[0]))
        for sol_idx in range(weights_mat.shape[0]):
            curr_sol_mat = weights_mat[sol_idx, :]
            accuracy[sol_idx], _ = predict_outputs(curr_sol_mat, data_inputs, data_outputs, activation=activation)
        return accuracy
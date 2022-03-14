#    I'm trying to create a class of neural networks
#
#    output = bn + wn * ... * f(b2 + w1 * f(b1 + w1 * input))
#        
#    I have to provide to the model constructor information about the layers
#    It have to be a list of amount of neurons in every layer, including an input and the output
#
#
#
#
#
import numpy as np
import matplotlib.pyplot as plt

# non-linear function(s?) and their(?) derivatives

def relu(x):
    return np.maximum(x, 0)

def d_relu(x):
    return (x >= 0).astype(float)

#
def correct_value(x):
    x = np.array(x)
    return x @ x.T

# basically my class

class NeuralNetwork:

    # Constructor
    def __init__(self, layers_list):

        #
        self.alpha = 0
        self.trainings = 0
        self.d_weights = []
        self.d_biases = []
        self.d_t = []
        self.d_h = []
        self.t = []
        self.h = []
        #

        self.err_list = []

        self.layers_list = layers_list

        self.weights = []
        self.biases = []

        for i in range(len(layers_list) - 1):
            self.weights.append(np.random.random((layers_list[i], layers_list[i + 1])))
            self.biases.append(np.random.random((layers_list[i + 1], 1)))

    # learning
    def train(self, alpha, trainings):

        #refreshing
        self.d_weights.clear()
        self.d_biases.clear()
        self.d_t.clear()
        self.d_h.clear()
        self.alpha = alpha
        self.trainings = trainings

        #learning cycle
        for i in range(self.trainings):

            input = np.random.random((self.layers_list[0], 1)).T
            y = correct_value(input)

            #forward
            output = self.calculate(input)

            #backward
            err = np.sum(np.linalg.matrix_power(output - y, 2))
            d_output = np.array([[2 * np.sum(output - y)]])

            final_d_h = d_output
            self.d_h.insert(0, final_d_h)

            final_d_t = d_output
            self.d_t.insert(0, final_d_t)

            final_d_b = final_d_t
            self.d_biases.insert(0, final_d_b)

            final_d_w = self.h[-2].T @ final_d_t
            self.d_weights.insert(0, final_d_w)

            for i in range(1, len(self.weights)):

                d_h = self.d_t[0] @ self.weights[-i].T
                self.d_h.insert(0, d_h)

                d_t = self.d_h[0] * d_relu(self.t[-i - 1])
                self.d_t.insert(0, d_t)

                d_b = d_t
                self.d_biases.insert(0, d_b)

                d_w = self.h[-i - 2].T @ self.d_t[0]
                self.d_weights.insert(0, d_w)

            #update
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] - self.d_weights[i] * self.alpha
                self.biases[i] = self.biases[i] - self.d_biases[i].T * self.alpha

            #add error
            self.err_list.append(err)

    #random input
    def show_random(self):

        input = np.random.random((self.layers_list[0], 1)).T
        output = self.calculate(input)
        y = correct_value(input)

        err = np.sum(np.linalg.matrix_power(output - y, 2))

        print('Neural network output: ' + str(output))
        print('Correct output: ' + str(y))
        print('Error: ' + str(err))

        plt.plot(self.err_list)
        plt.show()

    #determined input
    def show(self, input):

        output = self.calculate(input)
        y = correct_value(input)

        err = np.sum(np.linalg.matrix_power(output - y, 2))

        print('Neural network output: ' + str(output))
        print('Correct output: ' + str(y))
        print('Error: ' + str(err))

        plt.plot(self.err_list)
        plt.show()

    #forward
    def calculate(self, input):

        temp = np.array(input)
        self.t.clear()
        self.h.clear()
        self.t.append(input)
        self.h.append(input)

        for i in range(len(self.weights) - 1):
            self.t.append(temp @ self.weights[i] + self.biases[i].T)
            temp = relu(temp @ self.weights[i] + self.biases[i].T)
            self.h.append(temp)

        output = temp @ self.weights[len(self.weights) - 1] + self.biases[len(self.biases) - 1].T
        self.t.append(output)
        self.h.append(output)

        return output
        
layers = [3, 10, 7, 5, 1]
input = [1, 2, 3]
alpha = 0.001
trainings = 30000

model = NeuralNetwork(layers)
model.show(input)
model.train(alpha, trainings)
model.show(input)

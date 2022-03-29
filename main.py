#    I'm using this notation:
#
#    output = bn + wn * ... * f(b2 + w1 * f(b1 + w1 * input))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

#important: I use d_var as the name of derivative, not differential

#normalization function
def softmax(x):
    out = np.exp(x)
    return out / np.sum(out, axis=1, keepdims=True)
 #   expo = np.exp(x)
 #   expo_sum = np.sum(np.exp(x))
 #   return expo / expo_sum

#activation functions and their derivatives
def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def d_relu(x):
    return (x >= 0).astype(float)

def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def d_tanh(x):
    return 1 - tanh(x) ** 2

#loss functions and their derivatives
def mse(output, y):
    err = 0
    for j in range(len(output)):
        err += np.sum(np.square(output[j] - y[j]))
    return np.array(err)

#doesn't really work... I should learn about it more
def cross_entropy(output, y):
    return np.sum(-np.log(np.array([output[j, y[j]] for j in range(len(y))])))

def d_mse(output, y):
    d_error = []
    for j in range(len(output)):
        d_error.append(2 * (output[j] - y[j]))
    return np.array(d_error)

def d_cross_entropy(output, y):
    d_error = []
    for j in range(len(output)):
        d_error.append(2 * (output[j] - y[j]))
    return np.array(d_error)

class NeuralNetwork:

    #сonstructor
    def __init__(self, layers_list, activation, loss, normalize = None):

        self.alpha = 0
        self.current_training = 0
        self.d_weights = []
        self.d_biases = []
        self.d_t = []
        self.d_h = []
        self.t = []
        self.h = []
        self.test_err_list = []
        self.train_err_list = []
        self.weights = []
        self.biases = []
        self.activation = activation
        self.loss = loss
        self.normalize = normalize

        #optimal type of weights & biases initialization
        for i in range(len(layers_list) - 1):
            self.weights.append((np.random.rand(layers_list[i], layers_list[i + 1]) - 0.5) * 2 * np.sqrt(1 / layers_list[i]))
            self.biases.append((np.random.rand(layers_list[i + 1], 1) - 0.5) * 2 * np.sqrt(1 / layers_list[i]))
        
    #activation
    def __activation__(self, x):
        if self.activation == 'relu':
            return relu(x)
        elif self.activation == 'sigmoid':
            return sigmoid(x)
        elif self.activation == 'tanh':
            return tanh(x)

    def __d_activation__(self, x):
        if self.activation == 'relu':
            return d_relu(x)
        elif self.activation == 'sigmoid':
            return d_sigmoid(x)
        elif self.activation == 'tanh':
            return d_tanh(x)

    #error calculation
    def __error__(self, output, y):
        if self.loss == 'mse':
            return mse(output, y)
        elif self.loss == 'cross entropy':
            return cross_entropy(output, y)

    def __d_error_d_output__(self, output, y):
        if self.loss == 'mse':
            return d_mse(output, y)
        elif self.loss == 'cross entropy':
            return d_cross_entropy(output, y)

    #normalize
    def __normalize__(self, x):
        if self.normalize == 'softmax':
            return softmax(x)

    #functions for training
    def __train_refreshing__(self):
        self.d_weights.clear()
        self.d_biases.clear()
        self.d_t.clear()
        self.d_h.clear()
        self.train_err_list = []
        self.test_err_list = []

    def __backward__(self, output, y, err):

        d_output = self.__d_error_d_output__(output, y)

        final_d_h = d_output
        self.d_h.insert(0, final_d_h)

        final_d_t = d_output
        self.d_t.insert(0, final_d_t)

        final_d_b = np.sum(final_d_t, axis=0, keepdims = True)
        self.d_biases.insert(0, final_d_b)

        final_d_w = self.h[-2].T @ final_d_t
        self.d_weights.insert(0, final_d_w)

        for i in range(1, len(self.weights)):

            d_h = self.d_t[0] @ self.weights[-i].T
            self.d_h.insert(0, d_h)

            d_t = self.d_h[0] * self.__d_activation__(self.t[-i - 1])
            self.d_t.insert(0, d_t)

            d_b = np.sum(d_t, axis=0, keepdims = True)
            self.d_biases.insert(0, d_b)

            d_w = self.h[-i - 2].T @ self.d_t[0]
            self.d_weights.insert(0, d_w)

    def __update__(self):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - self.d_weights[i] * self.alpha
            self.biases[i] = self.biases[i] - self.d_biases[i].T * self.alpha

    #calculating output with providing optional parameters: list of t and h
    def __train_calculate__(self, input):

        temp = np.array(input)
        self.t.clear()
        self.h.clear()
        self.t.append(input)
        self.h.append(input)

        for i in range(len(self.weights) - 1):
            self.t.append(temp @ self.weights[i] + self.biases[i].T)
            temp = self.__activation__(temp @ self.weights[i] + self.biases[i].T)
            self.h.append(temp)

        if self.normalize is not None:
            output = self.__normalize__(temp @ self.weights[-1] + self.biases[-1].T)
        else:
            output = temp @ self.weights[-1] + self.biases[-1].T
        self.t.append(output)
        self.h.append(output)

        return output

    #learning
    def train(self, alpha, trainings, epochs, batch_size, dataset = None, test_dataset = None):

        #refreshing
        self.__train_refreshing__()

        #initializing
        self.alpha = alpha

        #learning cycle
        for i in range(epochs):

            random.shuffle(dataset)

            for j in range(trainings):

                self.current_training = j

                #input and correct output calculation
                batch_x, batch_y = zip(*dataset[j * batch_size : j * batch_size + batch_size])
                input = np.concatenate(batch_x, axis=0)
                y = []
                for i in batch_y:
                    y.append([i])
                y = np.array(batch_y)

                #forward
                output = self.__train_calculate__(input)
    
                #error calculation and appending to the error list

                #test data calculation
                if test_dataset is not None:
                    test_batch_x, test_batch_y = zip(*test_dataset[j * len(test_dataset) : j * len(test_dataset) + len(test_dataset)])
                    test_input = np.concatenate(test_batch_x, axis=0)
                    test_y = []
                    for i in test_batch_y:
                        test_y.append([i])
                    test_y = np.array(test_batch_y)
                    test_output = self.calculate(test_input)

                    #test error calculation and adding average error per item in the error list
                    err_test = self.__error__(test_output, test_y)
                    self.test_err_list.append(err_test / len(test_dataset))

                #train error calculation and adding average error per item in the error list
                err_train = self.__error__(output, y)
                self.train_err_list.append(err_train / len(dataset))

                #backward
                self.__backward__(output, y, err_train)

                #update
                self.__update__()

    #determined input test
    def show_determined_test(self, input = None, dataset_i = None):

        if dataset_i is not None:
            input = dataset_i[0]
            y = [dataset_i[1]]
            output = self.calculate(input)

            err = self.__error__(output, y)
        else:
            output = self.calculate(input)
            y = self.__correct_value__(input)

            err = self.__error__(output, y)

        print('Neural network output: ' + str(output))
        print('Correct output: ' + str(y))
        print('Error: ' + str(err))

    #average error
    def show_error(self):
        fig, ax = plt.subplots()
        ax.plot(self.train_err_list, label='Train error')
        ax.plot(self.test_err_list, label='Test error')
        ax.grid()
        ax.legend()
        plt.suptitle('Average error per item in batch')
        plt.xlabel('Trainings')
        plt.ylabel('Error')
        plt.show()

    #calculating output
    def calculate(self, input):

        temp = np.array(input)

        for i in range(len(self.weights) - 1):
            temp = self.__activation__(temp @ self.weights[i] + self.biases[i].T)

        if self.normalize is not None:
            output = self.__normalize__(temp @ self.weights[-1] + self.biases[-1].T)
        else:
            output = temp @ self.weights[-1] + self.biases[-1].T

        return output

from sklearn import datasets

'''
#iris

#obtaining dataset
iris = datasets.load_iris()
dataset = [(iris.data[i][None, ...], iris.target[i]) for i in range(len(iris.target))]
random.shuffle(dataset)

#using part of the original data to train our neural network
dataset_trainings_len = len(dataset) // 10 * 9
dataset_training = []
for i in range(dataset_trainings_len):
    dataset_training.append(dataset[i])

#and other part to test our network
test_dataset = []
for i in range(dataset_trainings_len, len(dataset)):
    test_dataset.append(dataset[i])

#providing some hyperparameters to the network
layers = [4, 10, 1]
alpha = 0.00005
batch_size = len(dataset_training)
trainings = len(dataset_training) //  batch_size
epochs = 1000

#creating and training the model
model = NeuralNetwork(layers)
model.train(alpha, trainings, epochs, batch_size, dataset_training, test_dataset)

#testing our model using test dataset
for i in range(10):
    model.show_determined_test(dataset_i = test_dataset[i])
    print()

model.show_error()

for i in range(3):
    print()
'''

#digits

#obtaining dataset
digits = datasets.load_digits()
arr_out = []
for i in range(10):
    out = []
    for j in range(10):
        if (i == j):
            out.append(1)
        else:
            out.append(0)
    arr_out.append(out)

dataset = [(digits.data[i][None, ...], arr_out[digits.target[i]]) for i in range(len(digits.target))]
random.shuffle(dataset)

#using part of the original data to train our neural network
dataset_trainings_len = len(dataset) // 10 * 9
dataset_training = []
for i in range(dataset_trainings_len):
    dataset_training.append(dataset[i])

#and other part to test our network
test_dataset = []
for i in range(dataset_trainings_len, len(dataset)):
    test_dataset.append(dataset[i])

#providing some hyperparameters to the network
layers = [64, 32, 16, 10]
alpha = 0.00003
batch_size = len(dataset_training)
trainings = len(dataset_training) //  batch_size
epochs = 1000

#creating and training the model
model = NeuralNetwork(layers, 'tanh', 'mse', 'softmax')
model.train(alpha, trainings, epochs, batch_size, dataset_training, test_dataset)

#testing our model using test dataset
for i in range(10):
    model.show_determined_test(dataset_i = test_dataset[i])
    print()

model.show_error()

for i in range(3):
    print()

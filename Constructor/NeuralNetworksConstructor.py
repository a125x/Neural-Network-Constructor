#    I'm using this notation:
#
#    output = bn + wn * ... * f(b2 + w1 * f(b1 + w1 * input))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

from numpy.core.fromnumeric import shape

#important: I use d_var as the name of derivative, not differential

#normalization function
def softmax(x):
    out = np.exp(x)
    return out / np.sum(out, axis=1, keepdims=True)

#activation functions and their derivatives
def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def tanh(x):
    return (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))

def d_relu(x):
    return (x >= 0).astype(float)

def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1-s)

def d_tanh(x):
    return 1 - tanh(x)**2

#loss functions and their derivatives
def mse(output, y):
    err = 0
    for j in range(len(output)):
        err += np.sum(np.square(output[j] - y[j]))
    return np.array(err)

def cross_entropy(output, y):
    return np.sum(-np.log(np.array([output[j, y[j]] for j in range(len(y))])))

def d_mse(output, y):
    d_error = []
    for j in range(len(output)):
        d_error.append(output[j]-y[j])
    return np.array(d_error)

#after softmax
def d_cross_entropy(output, y):
    d_error = []
    for j in range(len(output)):
        d_error.append(output[j]-y[j])
    return np.array(d_error)

class NeuralNetwork:

    #сonstructor
    def __init__(self, layers_list, activation, type):

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
        self.best_weights = []
        self.best_biases = []
        self.best_error = 0
        self.activation = activation
        self.type = type
        if type == 'classification':
            self.loss = 'cross entropy'
            self.normalize = 'softmax'
        elif type == 'regression':
            self.loss = 'mse'
            self.normalize = None

        #optimal type of weights & biases initialization
        for i in range(len(layers_list) - 1):
            self.weights.append(
                (np.random.rand(
                    layers_list[i], 
                    layers_list[i+1])
                 -0.5) * 2 * np.sqrt(1/layers_list[i]))
            self.biases.append(
                (np.random.rand(
                    layers_list[i+1], 
                    1) 
                 - 0.5) * 2 * np.sqrt(1/layers_list[i]))

        self.best_weights = self.weights
        self.best_biases = self.biases

    #file reading
    def read_weights(self, filename):

        #data preprocessing
        with open(filename, 'r') as file:
            data = file.read().split('\n')

        arrays = []
        for item in data:
            arrays.append(item.split())
            for j in range(len(arrays[-1])):
                try:
                    arrays[-1][j] = float(arrays[-1][j])
                except Exception:
                    continue

        count_matrix = -1 #because file starts with a hollow string
        count_rows = 0
        #because matrix length is half of the file, 
        #and other part is taken by biases
        for i in range(len(arrays) // 2):
            if arrays[i] == []:
                count_matrix += 1
                count_rows = 0
            else:
                self.weights[count_matrix][count_rows] = arrays[i]
                count_rows += 1

        count_biases = -1 #because weights divided from biases with a hollow string
        count_rows = 0
        #because matrix length is half of the file, 
        #and other part is taken by biases
        for i in range(len(arrays) // 2, len(arrays)):
            if arrays[i] == []:
                count_biases += 1
                count_rows = 0
            else:
                self.biases[count_biases][count_rows] = arrays[i]
                count_rows += 1

    def print_weights(self, filename):

        open(filename, 'w').close()

        for i in range(len(self.weights)):
            with open(filename, 'ab') as file:
                file.write(b'\n')
                np.savetxt(file, self.weights[i])
        
        for i in range(len(self.biases)):
            with open(filename, 'ab') as file:
                file.write(b'\n')
                np.savetxt(file, self.biases[i])
        
        
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

        final_d_b = np.sum(final_d_t, axis=0, keepdims=True)
        self.d_biases.insert(0, final_d_b)

        final_d_w = self.h[-2].T @ final_d_t
        self.d_weights.insert(0, final_d_w)

        for i in range(1, len(self.weights)):

            d_h = self.d_t[0] @ self.weights[-i].T
            self.d_h.insert(0, d_h)

            d_t = self.d_h[0] * self.__d_activation__(self.t[-i-1])
            self.d_t.insert(0, d_t)

            d_b = np.sum(d_t, axis=0, keepdims=True)
            self.d_biases.insert(0, d_b)

            d_w = self.h[-i-2].T @ self.d_t[0]
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
            self.t.append(
                temp @ self.weights[i] 
                + self.biases[i].T)
            temp = self.__activation__(
                temp @ self.weights[i] 
                + self.biases[i].T)
            self.h.append(temp)

        if self.normalize is not None:
            output = self.__normalize__(
                temp @ self.weights[-1] 
                + self.biases[-1].T)
        else:
            output = temp @ self.weights[-1] + self.biases[-1].T

        self.t.append(output)
        self.h.append(output)

        return output

    def __batch_from_data__(self, dataset, training_counter, batch_size):
        batch_x, batch_y = zip(
            *dataset[
                training_counter * batch_size : 
                training_counter * batch_size 
                + batch_size])
        input = np.concatenate(batch_x, axis=0)
        y = []
        for i in batch_y:
            y.append([i])
        y = np.array(batch_y)
        return input, y

    #learning
    def train(
        self, 
        alpha, 
        trainings, 
        epochs, 
        batch_size, 
        dataset = None, 
        test_dataset = None):

        #refreshing
        self.__train_refreshing__()

        #initializing
        self.alpha = alpha

        #learning cycle
        for epochs_counter in range(epochs):

            random.shuffle(dataset)

            for training_counter in range(trainings):

                self.current_training = training_counter

                #input and correct output calculation
                input, y = self.__batch_from_data__(dataset,
                                                   training_counter, 
                                                   batch_size)

                #forward
                output = self.__train_calculate__(input)
    
                #error calculation and appending to the error list

                #test data calculation
                if test_dataset is not None:
                    if self.type == 'regression':
                        test_input, test_y = self.__batch_from_data__(
                            test_dataset,
                            training_counter,
                            len(test_dataset) // trainings
                            )

                        #forward
                        test_output = self.calculate(test_input)

                        #test error calculation and adding 
                        #average error per item in the error list
                        err_test = self.__error__(test_output, test_y)
                        self.test_err_list.append(
                            err_test / len(test_dataset))

                        #finding lowest error per item in the test batch
                        #and changing best weights
                        if epochs_counter == 0 and training_counter == 0:
                            self.best_error = self.test_err_list[0]
                        elif self.test_err_list[-1] < self.best_error:
                            self.best_error = self.test_err_list[-1]
                            self.best_weights = self.weights
                            self.best_biases = self.biases

                    elif self.type == 'classification':
                        self.test_err_list.append(self.accuracy(test_dataset))

                        #finding best accuracy
                        #and changing best weights
                        if epochs_counter == 0 and training_counter == 0:
                            self.best_error = self.test_err_list[0]
                        elif self.test_err_list[-1] >= self.best_error:
                            self.best_error = self.test_err_list[-1]
                            self.best_weights = self.weights
                            self.best_biases = self.biases
                            '''
                        else:
                           print('best accuracy: ',
                                self.best_error,
                               ', current accuracy: ',
                              self.test_err_list[-1])
                           '''
                #train error calculation and adding 
                #average error per item in the error list
                err_train = self.__error__(output, y)
                if self.type == 'regression':
                    self.train_err_list.append(err_train / len(dataset))
                elif self.type == 'classification':
                    self.train_err_list.append(self.accuracy(dataset))

                #backward
                self.__backward__(output, y, err_train)

                #update
                self.__update__()

    #prediction accuracy calculating
    def accuracy(self, dataset, mode='last'):
        
        correct_answers = 0
        for x_test, y_test in dataset:
            if mode == 'last':
                out_test = self.calculate(x_test)
            elif mode == 'best':
                out_test = self.calculate(x_test, mode='best')
            y_pred = np.argmax(out_test)
            if y_pred == np.argmax(y_test):
                correct_answers += 1

        return correct_answers / len(dataset)

    #determined input test
    def show_determined_test(self, input = None, dataset_i = None):

        if self.type == 'classification':

            if dataset_i is not None:
                input = dataset_i[0]
                y = [dataset_i[1]]
                output = self.calculate(input)

                err = self.__error__(output, y)
                
                y = np.argmax(y)
                output = np.argmax(output)
            else:
                output = self.calculate(input)
                y = self.__correct_value__(input)

                err = self.__error__(output, y)
                
                y = np.argmax(y)
                output = np.argmax(output)

            print('Neural network output (cleared): ' + str(output))
            print('Correct output (cleared): ' + str(y))
            print('Cross-entropy before clearing: ' + str(err))

        elif self.type == 'regression':

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
        if self.type == 'regression':
            ax.plot(self.train_err_list, label='Train error')
            ax.plot(self.test_err_list, label='Test error')
            plt.suptitle('Average error per item in batch')
            plt.xlabel('Trainings')
            plt.ylabel('Error')
        elif self.type == 'classification':
            ax.plot(self.train_err_list, label='Train accuracy')
            ax.plot(self.test_err_list, label='Test accuracy')
            plt.suptitle('Prediction accuracy in batch')
            plt.xlabel('Trainings')
            plt.ylabel('Accuracy')

        ax.grid()
        ax.legend()
        plt.show()

    #calculating output
    def calculate(self, input, mode='last'):

        temp = np.array(input)

        if mode == 'last':
            for i in range(len(self.weights) - 1):
                temp = self.__activation__(
                    temp @ self.weights[i] 
                    + self.biases[i].T)

            if self.normalize is not None:
                output = self.__normalize__(
                    temp @ self.weights[-1] 
                    + self.biases[-1].T)
            else:
                output = temp @ self.weights[-1] + self.biases[-1].T

        elif mode == 'best':
            for i in range(len(self.best_weights) - 1):
                temp = self.__activation__(
                    temp @ self.best_weights[i] 
                    + self.best_biases[i].T)

            if self.normalize is not None:
                output = self.__normalize__(
                    temp @ self.best_weights[-1] 
                    + self.best_biases[-1].T)
            else:
                output = temp @ self.best_weights[-1] + self.best_biases[-1].T

        return output
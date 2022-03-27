import numpy as np
import matplotlib.pyplot as plt
import random

class NeuralNetwork:

    #сonstructor
    def __init__(self, layers_list):

        self.alpha = 0
        self.trainings = 0
        self.epochs = 0
        self.current_training = 0
        self.d_weights = []
        self.d_biases = []
        self.d_t = []
        self.d_h = []
        self.t = []
        self.h = []
        self.dataset = []
        self.train_err_list = []
        self.layers_list = layers_list
        self.weights = []
        self.biases = []

        #optimal type of weights & biases initialization
        for i in range(len(layers_list) - 1):
            self.weights.append((np.random.rand(layers_list[i], layers_list[i + 1]) - 0.5) * 2 * np.sqrt(1 / layers_list[i]))
            self.biases.append((np.random.rand(layers_list[i + 1], 1) - 0.5) * 2 * np.sqrt(1 / layers_list[i]))

    #error calculation
    def __err_calc__(self, output, y):
        err = 0
        for j in range(len(output)):
            err = err + np.sum(np.square(output[j] - y[j]))
        return np.array(err)

    def __d_output_calc__(self, output, y):
        d_out = []
        for j in range(len(output)):
            d_out.append(2 * (output[j] - y[j]))
        return np.array(d_out)

    #functions for training
    def __train_refreshing__(self):
        self.d_weights.clear()
        self.d_biases.clear()
        self.d_t.clear()
        self.d_h.clear()
        self.alpha = alpha

    def __backward__(self, output, y, err):

        d_output = self.__d_output_calc__(output, y)

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

            d_t = self.d_h[0] * self.__d_relu__(self.t[-i - 1])
            self.d_t.insert(0, d_t)

            d_b = np.sum(d_t, axis=0, keepdims=True)
            self.d_biases.insert(0, d_b)

            d_w = self.h[-i - 2].T @ self.d_t[0]
            self.d_weights.insert(0, d_w)

    def __update__(self):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - self.d_weights[i] * self.alpha
            self.biases[i] = self.biases[i] - self.d_biases[i].T * self.alpha

    def __train_input__(self):
        return np.random.random((self.layers_list[0], 1)).T

    def __train_correct_value__(self, input):
        return dataset[self.current_training][1]

    #calculating output with providing optional parameters
    def __train_calculate__(self, input):

        temp = np.array(input)
        self.t.clear()
        self.h.clear()
        self.t.append(input)
        self.h.append(input)

        for i in range(len(self.weights) - 1):
            self.t.append(temp @ self.weights[i] + self.biases[i].T)
            temp = self.__relu__(temp @ self.weights[i] + self.biases[i].T)
            self.h.append(temp)

        output = temp @ self.weights[len(self.weights) - 1] + self.biases[len(self.biases) - 1].T
        self.t.append(output)
        self.h.append(output)

        return output

    #activation functions
    def __relu__(self, x):
        return np.maximum(x, 0)

    def __d_relu__(self, x):
        return (x >= 0).astype(float)

    #learning
    def train(self, alpha, trainings, epochs, batch_size, dataset = None):

        #refreshing
        self.__train_refreshing__()

        #initializing
        self.trainings = trainings
        self.epochs = epochs
        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset = []

        #learning cycle
        for i in range(self.epochs):

            random.shuffle(dataset)

            for j in range(self.trainings):

                self.current_training = j

                #input and correct output calculation
                if dataset is not None:
                    batch_x, batch_y = zip(*dataset[j * batch_size : j * batch_size + batch_size])
                    input = np.concatenate(batch_x, axis=0)
                    y = []
                    for i in batch_y:
                        y.append([i])
                    y = np.array(batch_y)
                else:
                    input = self.__train_input__()
                    y = self.__train_correct_value__(input)

                #forward
                output = self.__train_calculate__(input)
    
                #error calculation and appending to the error list
                err = self.__err_calc__(output, y)
                self.train_err_list.append(err)

                #backward
                self.__backward__(output, y, err)

                #update
                self.__update__()

    #determined input test
    def show_determined_test(self, input = None, dataset_i = None):

        if dataset_i is not None:
            input = dataset_i[0]
            y = [dataset_i[1]]
            output = self.calculate(input)

            err = self.__err_calc__(output, y)
        else:
            output = self.calculate(input)
            y = self.__correct_value__(input)

            err = self.__err_calc__(output, y)

        print('Neural network output: ' + str(output))
        print('Correct output: ' + str(y))
        print('Error: ' + str(err))

    def show_error(self):
        plt.plot(self.train_err_list)
        plt.show()

    #calculating output
    def calculate(self, input):

        temp = np.array(input)

        for i in range(len(self.weights) - 1):
            temp = self.__relu__(temp @ self.weights[i] + self.biases[i].T)

        output = temp @ self.weights[len(self.weights) - 1] + self.biases[len(self.biases) - 1].T

        return output

from sklearn import datasets


#iris
iris = datasets.load_iris()
dataset = [(iris.data[i][None, ...], iris.target[i]) for i in range(len(iris.target))]
random.shuffle(dataset)

dataset_trainings_len = len(dataset) // 10 * 9
dataset_training = []
for i in range(dataset_trainings_len):
    dataset_training.append(dataset[i])

dataset_test = []
for i in range(dataset_trainings_len, len(dataset)):
    dataset_test.append(dataset[i])

layers = [4, 10, 1]
alpha = 0.0002
batch_size = len(dataset_training)
trainings = len(dataset_training) //  batch_size
epochs = 1000

model = NeuralNetwork(layers)
model.train(alpha, trainings, epochs, batch_size, dataset_training)

for i in range(10):
    model.show_determined_test(dataset_i = dataset_test[i])
    print()

model.show_error()

for i in range(3):
    print()

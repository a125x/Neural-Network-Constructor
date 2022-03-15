import matplotlib.pyplot as plt

class NeuralNetwork:

    #сonstructor
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

    #functions for training
    def __train_refreshing__(self):
        self.d_weights.clear()
        self.d_biases.clear()
        self.d_t.clear()
        self.d_h.clear()
        self.alpha = alpha
        self.trainings = trainings

    def __backward__(self, output, y, err):

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

            d_t = self.d_h[0] * self.__d_relu__(self.t[-i - 1])
            self.d_t.insert(0, d_t)

            d_b = d_t
            self.d_biases.insert(0, d_b)

            d_w = self.h[-i - 2].T @ self.d_t[0]
            self.d_weights.insert(0, d_w)

    def __update__(self):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - self.d_weights[i] * self.alpha
            self.biases[i] = self.biases[i] - self.d_biases[i].T * self.alpha

    #ATTENTION: change random input back after experiments
    def __random_input__(self):
        input = 10 * np.random.random((self.layers_list[0], 1)).T
        return input

    def __correct_value__(self, input):
        input = np.array(input)
        return input @ input.T

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
    def train(self, alpha, trainings):

        #refreshing
        self.__train_refreshing__()

        #learning cycle
        for i in range(self.trainings):

            #input and correct output calculation
            input = self.__random_input__()
            y = self.__correct_value__(input)

            #forward
            output = self.__train_calculate__(input)

            #error calculation and appending to the error list
            err = np.sum(np.linalg.matrix_power(output - y, 2))
            self.err_list.append(err)

            #backward
            self.__backward__(output, y, err)

            #update
            self.__update__()

    #random input
    def show_random_test(self):

        input = np.random.random((self.layers_list[0], 1)).T
        output = self.calculate(input)
        y = self.__correct_value__(input)

        err = np.sum(np.linalg.matrix_power(output - y, 2))

        print('Neural network output: ' + str(output))
        print('Correct output: ' + str(y))
        print('Error: ' + str(err))

    #determined input
    def show_determined_test(self, input):

        output = self.calculate(input)
        y = self.__correct_value__(input)

        err = np.sum(np.linalg.matrix_power(output - y, 2))

        print('Neural network output: ' + str(output))
        print('Correct output: ' + str(y))
        print('Error: ' + str(err))

    def show_error(self):
        plt.plot(self.err_list)
        plt.show()

    #calculating output
    def calculate(self, input):

        temp = np.array(input)

        for i in range(len(self.weights) - 1):
            temp = self.__relu__(temp @ self.weights[i] + self.biases[i].T)

        output = temp @ self.weights[len(self.weights) - 1] + self.biases[len(self.biases) - 1].T

        return output
  
#in this particular case I have some interesting results: network in attempts to optimize
#it's behavior stuck at the only one specific output

layers = [2, 4, 4, 4, 1]
input = [3, 5]
alpha = 0.01
trainings = 20000

model = NeuralNetwork(layers)
model.train(alpha, trainings)

for i in range(10):
    model.show_random_test()

print()

model.show_determined_test(input)

model.show_error()

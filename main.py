import Class.NeuralNetworksClass as nnc
import Tests.NeuralNetworksTests
from sklearn import datasets
import random

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

dataset = [
    (digits.data[i][None, ...], 
     arr_out[digits.target[i]]) 
    for i in range(len(digits.target))]
random.shuffle(dataset)

#using part of the original data to train our neural network
dataset_trainings_len = len(dataset) // 100 * 95
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
epochs = 200

#creating and training the model
model = nnc.NeuralNetworks(layers, 'tanh', 'classification')
model.train(
    alpha, 
    trainings, 
    epochs, 
    batch_size, 
    dataset_training, 
    test_dataset)

#testing our model using test dataset
for i in range(10):
    model.show_determined_test(dataset_i = test_dataset[i])
    print()

model.show_error()

for i in range(3):
    print()
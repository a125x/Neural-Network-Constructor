import pytest
import numpy as np
from Constructor import NeuralNetworksConstructor as nnc
from sklearn import datasets, preprocessing
import random
import matplotlib.pyplot as plt
from azureml.opendatasets import MNIST

def test_mnist_numbers():
    mnist = MNIST.get_tabular_dataset()
    mnist_df = mnist.to_pandas_dataframe()
    mnist_df.info()

    mnist_train = MNIST.get_tabular_dataset(dataset_filter='train')
    mnist_train_df = mnist_train.to_pandas_dataframe()
    X_train = mnist_train_df.drop("label", axis=1).astype(int).values/255.0
    x_train = []
    for i in range(len(X_train)):
        x_train.append([X_train[i]])
    Y_train = mnist_train_df.filter(items=["label"]).astype(int).values
    y_train = [item for sublist in Y_train for item in sublist]

    mnist_test = MNIST.get_tabular_dataset(dataset_filter='test')
    mnist_test_df = mnist_test.to_pandas_dataframe()
    X_test = mnist_test_df.drop("label", axis=1).astype(int).values/255.0
    x_test = []
    for i in range(len(X_test)):
        x_test.append([X_test[i]])
    Y_test = mnist_test_df.filter(items=["label"]).astype(int).values
    y_test = [item for sublist in Y_test for item in sublist]
    
    classes = 10
    arr_out = np.eye(classes, dtype=int)
    
    dataset_training = [
        (x_train[i],  
        arr_out[y_train[i]]) 
        for i in range(len(x_train))]
    random.shuffle(dataset_training)
    
    test_dataset = [
        (x_test[i],  
        arr_out[y_test[i]]) 
        for i in range(len(x_test))]
    random.shuffle(test_dataset)
    
    #providing some hyperparameters to the network
    layers = [784, 250, 50, 10]
    alpha = 0.00005
    batch_size = 300
    trainings = len(dataset_training) //  batch_size
    epochs = 10

    #creating and training the model
    model = nnc.NeuralNetwork(layers, 'tanh', 'classification')
    model.read_weights('Weights-Data\mnist_digits.txt')
    model.train(alpha, 
                trainings, 
                epochs, 
                batch_size, 
                dataset_training, 
                test_dataset)

    #testing our model using test dataset
    for i in range(10):
        model.show_determined_test(dataset_i = test_dataset[i])
        print()

    model.print_weights('Weights-Data\mnist_digits.txt')
    model2 = nnc.NeuralNetwork(layers, 'tanh', 'classification')
    model2.read_weights('Weights-Data\mnist_digits.txt')
    print('Trained model accuracy:', model.accuracy(test_dataset))
    print('Copied model accuracy:', model2.accuracy(test_dataset))
    model.print_weights('Weights-Data\mnist_digits.txt', mode='best')
    model2.read_weights('Weights-Data\mnist_digits.txt')
    print('Trained model best accuracy:', model.accuracy(test_dataset, mode='best'))
    print('Copied model best accuracy:', model2.accuracy(test_dataset))
    model.show_error()
    
#to prove is current model even working
#based on the toy datasets from sklearn
def test_iris_classification():

    #obtaining dataset
    classes = 3
    iris = datasets.load_iris()
    arr_out = np.eye(classes, dtype=int)

    dataset = [
        (iris.data[i][None, ...], 
         arr_out[iris.target[i]]) 
        for i in range(len(iris.target))]
    random.shuffle(dataset)

    #using part of the original data to train our neural network
    dataset_trainings_len = len(dataset) * 90 // 100
    dataset_training = []
    for i in range(dataset_trainings_len):
        dataset_training.append(dataset[i])

    #and other part to test our network
    test_dataset = []
    for i in range(dataset_trainings_len, len(dataset)):
        test_dataset.append(dataset[i])

    #providing some hyperparameters to the network
    layers = [4, 10, 3]
    alpha = 0.00003
    batch_size = len(dataset_training)
    trainings = len(dataset_training) //  batch_size
    epochs = 1000

    #creating and training the model
    model = nnc.NeuralNetwork(layers, 'tanh', 'classification')
    model.train(alpha, 
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

def test_numbers_classification():

    #obtaining dataset
    classes = 10
    digits = datasets.load_digits()
    arr_out = np.eye(classes, dtype=int)

    dataset = [
        (digits.data[i][None, ...], 
         arr_out[digits.target[i]]) 
        for i in range(len(digits.target))]
    random.shuffle(dataset)

    #using part of the original data to train our neural network
    dataset_trainings_len = len(dataset) * 90 // 100
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
    model = nnc.NeuralNetwork(layers, 'tanh', 'classification')
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

def test_diabetes_regression():

    #obtaining dataset
    diabetes = datasets.load_diabetes()
    #standartization
    diabetes.data = preprocessing.normalize(diabetes.data, axis=0)
    dataset = [
        (diabetes.data[i][None, ...], 
         diabetes.target[i]) for i in range(len(diabetes.target))]
    random.shuffle(dataset)

    #using part of the original data to train our neural network
    dataset_trainings_len = len(dataset) * 90 // 100
    dataset_training = []
    for i in range(dataset_trainings_len):
        dataset_training.append(dataset[i])

    #and other part to test our network
    test_dataset = []
    for i in range(dataset_trainings_len, len(dataset)):
        test_dataset.append(dataset[i])

    #providing some hyperparameters to the network
    layers = [10, 10, 1]
    alpha = 0.0000035
    batch_size = len(dataset_training)
    trainings = len(dataset_training) //  batch_size
    epochs = 2000

    #creating and training the model
    model = nnc.NeuralNetwork(layers, 'relu', 'regression')
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

def test_boston_regression():

    #obtaining dataset
    boston = datasets.load_boston()
    #standartization
    boston.data = preprocessing.normalize(boston.data, axis=0)
    dataset = [
        (boston.data[i][None, ...], 
         boston.target[i]) for i in range(len(boston.target))]
    random.shuffle(dataset)

    #using part of the original data to train our neural network
    dataset_trainings_len = len(dataset) * 90 // 100
    dataset_training = []
    for i in range(dataset_trainings_len):
        dataset_training.append(dataset[i])

    #and other part to test our network
    test_dataset = []
    for i in range(dataset_trainings_len, len(dataset)):
        test_dataset.append(dataset[i])

    #providing some hyperparameters to the network
    layers = [13, 10, 1]
    alpha = 0.000007
    batch_size = len(dataset_training)
    trainings = len(dataset_training) //  batch_size
    epochs = 5000

    #creating and training the model
    model = nnc.NeuralNetwork(layers, 'relu', 'regression')
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

def test_linnerud_regression():

    #obtaining dataset
    linnerud = datasets.load_linnerud()
    #standartization
    linnerud.data = preprocessing.normalize(linnerud.data, axis=0)
    dataset = [
        (linnerud.data[i][None, ...], 
         linnerud.target[i]) for i in range(len(linnerud.target))]
    random.shuffle(dataset)

    #using part of the original data to train our neural network
    dataset_trainings_len = len(dataset) * 80 // 100
    dataset_training = []
    for i in range(dataset_trainings_len):
        dataset_training.append(dataset[i])

    #and other part to test our network
    test_dataset = []
    for i in range(dataset_trainings_len, len(dataset)):
        test_dataset.append(dataset[i])

    #providing some hyperparameters to the network
    layers = [3, 10, 3]
    alpha = 0.000007
    batch_size = len(dataset_training)
    trainings = len(dataset_training) //  batch_size
    epochs = 5000

    #creating and training the model
    model = nnc.NeuralNetwork(layers, 'relu', 'regression')
    model.train(
        alpha, 
        trainings, 
        epochs, 
        batch_size, 
        dataset_training, 
        test_dataset)

    #testing our model using test dataset
    for i in range(len(test_dataset)):
        model.show_determined_test(dataset_i = test_dataset[i])
        print()

    model.show_error()

    for i in range(3):
        print()

def test_wine_classification():

    #obtaining dataset
    classes = 3
    wine = datasets.load_wine()
    arr_out = np.eye(classes, dtype=int)

    dataset = [
        (wine.data[i][None, ...], 
         arr_out[wine.target[i]]) 
        for i in range(len(wine.target))]
    random.shuffle(dataset)

    #using part of the original data to train our neural network
    dataset_trainings_len = len(dataset) * 80 // 100
    dataset_training = []
    for i in range(dataset_trainings_len):
        dataset_training.append(dataset[i])

    #and other part to test our network
    test_dataset = []
    for i in range(dataset_trainings_len, len(dataset)):
        test_dataset.append(dataset[i])

    #providing some hyperparameters to the network
    layers = [13, 30, 30, 15, 7, 3]
    alpha = 0.000002
    batch_size = len(dataset_training)
    trainings = len(dataset_training) //  batch_size
    epochs = 4000

    #creating and training the model
    model = nnc.NeuralNetwork(layers, 'relu', 'classification')
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

def test_breast_cancer_classification():

    #obtaining dataset
    classes = 2
    breast_cancer = datasets.load_breast_cancer()
    arr_out = np.eye(classes, dtype=int)

    dataset = [
        (breast_cancer.data[i][None, ...], 
         arr_out[breast_cancer.target[i]]) 
        for i in range(len(breast_cancer.target))]
    random.shuffle(dataset)

    #using part of the original data to train our neural network
    dataset_trainings_len = len(dataset) * 90 // 100
    dataset_training = []
    for i in range(dataset_trainings_len):
        dataset_training.append(dataset[i])

    #and other part to test our network
    test_dataset = []
    for i in range(dataset_trainings_len, len(dataset)):
        test_dataset.append(dataset[i])

    #providing some hyperparameters to the network
    layers = [30, 15, 10, 2]
    alpha = 0.000002
    batch_size = len(dataset_training)
    trainings = len(dataset_training) //  batch_size
    epochs = 1000

    #creating and training the model
    model = nnc.NeuralNetwork(layers, 'relu', 'classification')
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

#saving & reading weights from a txt file
def test_print_and_read_weights():

    #obtaining dataset
    classes = 3
    wine = datasets.load_wine()
    arr_out = np.eye(classes, dtype=int)

    dataset = [
        (wine.data[i][None, ...], 
         arr_out[wine.target[i]]) 
        for i in range(len(wine.target))]
    random.shuffle(dataset)

    #using part of the original data to train our neural network
    dataset_trainings_len = len(dataset) * 80 // 100
    dataset_training = []
    for i in range(dataset_trainings_len):
        dataset_training.append(dataset[i])

    #and other part to test our network
    test_dataset = []
    for i in range(dataset_trainings_len, len(dataset)):
        test_dataset.append(dataset[i])

    #providing some hyperparameters to the network
    layers = [13, 30, 30, 15, 7, 3]
    alpha = 0.00001
    batch_size = len(dataset_training)
    trainings = len(dataset_training) //  batch_size
    epochs = 400

    #creating and training the model
    model = nnc.NeuralNetwork(layers, 'relu', 'classification')
    
    model.train(
        alpha, 
        trainings, 
        epochs, 
        batch_size, 
        dataset_training, 
        test_dataset)

    model.print_weights('Weights-Data\data.txt')

    #testing our model using test dataset
    #model.show_error()
    print('Trained model accuracy:', model.accuracy(test_dataset))
    
    #creating the second model and obtaining weights for it
    model2 = nnc.NeuralNetwork(layers, 'relu', 'classification')
    model2.read_weights('Weights-Data\data.txt')

    #testing our model using test dataset
    print('Copied model accuracy:', model2.accuracy(test_dataset))

    print('Trained model best accuracy:', model.accuracy(test_dataset, mode='best'))

    model.print_weights('Weights-Data\data.txt', mode='best')
    
    model2.read_weights('Weights-Data\data.txt')

    #testing our model using test dataset
    print('Copied model best accuracy:', model2.accuracy(test_dataset))

#saving weights with the best results on the test dataset
def test_best_weights():

    #obtaining dataset
    classes = 3
    wine = datasets.load_wine()
    arr_out = np.eye(classes, dtype=int)

    dataset = [
        (wine.data[i][None, ...], 
         arr_out[wine.target[i]]) 
        for i in range(len(wine.target))]
    random.shuffle(dataset)

    #using part of the original data to train our neural network
    dataset_trainings_len = len(dataset) * 80 // 100
    dataset_training = []
    for i in range(dataset_trainings_len):
        dataset_training.append(dataset[i])

    #and other part to test our network
    test_dataset = []
    for i in range(dataset_trainings_len, len(dataset)):
        test_dataset.append(dataset[i])

    #providing some hyperparameters to the network
    layers = [13, 30, 30, 15, 7, 3]
    alpha = 0.00001
    batch_size = len(dataset_training)
    trainings = len(dataset_training) //  batch_size
    epochs = 400

    #creating and training the model
    model = nnc.NeuralNetwork(layers, 'relu', 'classification')

    for i in range(3):
        model.train(
            alpha, 
            trainings, 
            epochs, 
            batch_size, 
            dataset_training, 
            test_dataset)
        
        #testing our latest model using test dataset
        print('Latest model accuracy: ',
              model.accuracy(test_dataset))

        #testing our best model using test dataset
        print('Model with the best accuracy on the tests data accuracy: ', 
              model.accuracy(test_dataset, 'best'))
        
        #print(model.accuracy(dataset) != model.accuracy(dataset, 'best'))
        #print(model.best_weights == model.weights and model.best_biases == model.biases)
        #model.show_error()

#different batches length, tests: 
#__batch_from_data__(self, dataset, training_counter, batch_size)
#based on the digits classification toy dataset from sklearn
def test_batch_sizes():

    #obtaining dataset
    classes = 10
    digits = datasets.load_digits()
    arr_out = np.eye(classes, dtype=int)

    dataset = [
        (digits.data[i][None, ...], 
         arr_out[digits.target[i]]) 
        for i in range(len(digits.target))]
    random.shuffle(dataset)

    #using part of the original data to train our neural network
    dataset_trainings_len = len(dataset) * 90 // 100
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

    for i in range(1, 5):
        batch_size = i * 64
        trainings = len(dataset_training) //  batch_size
        epochs = 100 // trainings
    
        #creating and training the model
        model = nnc.NeuralNetwork(layers, 'tanh', 'classification')
        model.train(
            alpha, 
            trainings, 
            epochs, 
            batch_size, 
            dataset_training, 
            test_dataset)

        #testing our model using test dataset
        model.show_determined_test(dataset_i = test_dataset[i])
        print()

        model.show_error()

#test weights
def test_brbrbrb():
    #obtaining dataset
    classes = 3
    wine = datasets.load_wine()
    arr_out = np.eye(classes, dtype=int)

    dataset = [
        (wine.data[i][None, ...], 
         arr_out[wine.target[i]]) 
        for i in range(len(wine.target))]
    random.shuffle(dataset)

    #using part of the original data to train our neural network
    dataset_trainings_len = len(dataset) * 80 // 100
    dataset_training = []
    for i in range(dataset_trainings_len):
        dataset_training.append(dataset[i])

    #and other part to test our network
    test_dataset = []
    for i in range(dataset_trainings_len, len(dataset)):
        test_dataset.append(dataset[i])

    #providing some hyperparameters to the network
    layers = [13, 30, 30, 15, 7, 3]
    alpha = 0.00001
    batch_size = len(dataset_training)
    trainings = len(dataset_training) //  batch_size
    epochs = 3000
    
    model = nnc.NeuralNetwork(layers, 'relu', 'classification')
    model2 = nnc.NeuralNetwork(layers, 'relu', 'classification')

    for i in range(1):
        model.train(
            alpha, 
            trainings, 
            epochs, 
            batch_size, 
            dataset_training, 
            test_dataset)

        model.print_weights('Weights-Data\wine_weights.txt')

        #testing our model using test dataset
    
        print('Trained model best accuracy:', model.accuracy(test_dataset, mode='best'))
 
        model.print_weights('Weights-Data\wine_weights.txt', mode='best')
        model2.read_weights('Weights-Data\wine_weights.txt')

        #testing our model using test dataset
        print('Copied model best accuracy:', model2.accuracy(test_dataset))

        model.show_error()
    
def test_dropout():

    #obtaining dataset
    classes = 3
    wine = datasets.load_wine()
    arr_out = np.eye(classes, dtype=int)

    dataset = [
        (wine.data[i][None, ...], 
         arr_out[wine.target[i]]) 
        for i in range(len(wine.target))]
    random.shuffle(dataset)

    #using part of the original data to train our neural network
    dataset_trainings_len = len(dataset) * 80 // 100
    dataset_training = []
    for i in range(dataset_trainings_len):
        dataset_training.append(dataset[i])

    #and other part to test our network
    test_dataset = []
    for i in range(dataset_trainings_len, len(dataset)):
        test_dataset.append(dataset[i])

    #providing some hyperparameters to the network
    layers = [13, 30, 30, 15, 7, 3]
    alpha = 0.000002
    batch_size = len(dataset_training)
    trainings = len(dataset_training) //  batch_size
    epochs = 400

    #creating and training the model
    model = nnc.NeuralNetwork(layers, 'relu', 'classification')
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
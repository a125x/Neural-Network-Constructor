import numpy as np
from Constructor import NeuralNetworksConstructor as nnc
from sklearn import datasets, preprocessing
import random

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
def test_print_and_read_weights_iris():

    #obtaining dataset
    classes = 3
    iris = datasets.load_iris()
    arr_out = np.eye(classes, dtype=int)

    dataset = [
        (iris.data[i][None, ...], arr_out[iris.target[i]]) 
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
    model.train(alpha, trainings, epochs, batch_size, dataset_training, test_dataset)

    model.print_weights('Weights-Data\output.txt')

    #testing our model using test dataset
    model.show_error()
    print('Trained model accuracy:', model.accuracy(test_dataset))
    
    #creating the second model and obtaining weights for it
    model2 = nnc.NeuralNetwork(layers, 'tanh', 'classification')
    model2.read_weights('Weights-Data\output.txt')

    #testing our model using test dataset
    print('Copied model accuracy:', model2.accuracy(dataset))

#saving weights with the best results on the test dataset
def test_best_weights():

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

    for i in range(1):
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
        print(model.best_weights == model.weights)
        model.show_error()

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

#

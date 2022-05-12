# Neural-Network-Constructor
I mastered methods of creating and training neural networks, 
and in the process created a constructor that allows you to work
with multilayer perceptrons.

Final goal of the project is to allow user to create a basic neural 
networks, train them and use them. Networks must be used for 
classification or regression, user also can choose from different 
activation functions but only one for the whole network and the 
last layer's activation function is determinied by network's 
purpose. User also can choose hyperparameters for training and
shape of the whole network, but it shouldn't be too large, because 
constructor is not optimized and using mostly basic python.

Also constructor have some features such as saving weights which 
performed the best on the test data during trainings and saving 
weights and biases data in the text file and reading them into, 
well, network's weights itself. And of course constructor have a 
lot of functions to check the results of user's training.

This project based on my previous work: 
https://github.com/A125X/A-Simple-Neural-Networl

And this project: 
https://github.com/dkorobchenko-nv/nn-python



## So... How to use it?


First of all, you should prepare the train and test dataset. 
Each of them must be structured this way:
```
    [[input list for example 1], [output list for example 1]]
    [[input list for example 2], [output list for example 2]]
    and so on.
```

And after this shuffle them.

Secondly, provide hyperparameters to create the network.
It will be:

layers in this form:
```
    [amount of input neurons, hidden neurons1, ..., output neurons]
```

activation function:
```
    'tanh', 'sigmoid' or 'relu'
```
as a string and

type:
```
   'regression' or 'classification'
```
based on the task which you want to solve.

And after you can build a model:

```
from Constructor import NeuralNetworksConstructor as nnc

model = nnc.NeuralNetwork(layers, activation function, type)
```

Next, for training you have to 
provide hyperparameters for the training, such as:

```
training coefficient (as a float number), 
amount of trainings 
(usually len(dataset_training) //  batch_size, positive integer), 
amount of epochs (positive integer), batch size (positive integer), 
dataset for training and dataset for testing 
which you prepared at the first step.
```
And you can train your model like this:

```
model.train(
        alpha, 
        trainings, 
        epochs, 
        batch_size, 
        dataset = None, 
        test_dataset = None)
```

## Methods of providing information

After training you may like to see the results. 

For this I created a couple of pretty nice methods.

```
calculate(input, mode='last'): 

calculating the result fron the given input. 
Can be used for the predictions, input must be the same size
(in neurons) as all the training and test input examples in the
test and training datasets.
```
```
accuracy(dataset, mode='last'):

ONLY FOR CLASSIFICATION MODELS 
providing an accuracy of prediction based on the
given dataset.
```
```
show_error(): 

showing error/accuracy during trainings.
```
```
show_determined_test(input = None, dataset_i = None):

showing correct result based on the provided dataset item.
```

## Saving methods

```
print_weights(filename):

printing weidhts and biases in the txt file, firstly 
weights, secondary biases using standart nympy savetxt function
with rewriting previous data in this file
```

```
read_weights(filename):

setting weights & biases to those which written in the file 
previously by the print_weights method
```
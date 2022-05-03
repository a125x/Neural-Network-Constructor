# Neural-Network-Constructor
I mastered methods of creating and training neural networks, 
and in the process created a constructor that allows you to work
with multilayer perceptrons.

This project based on my previous work: https://github.com/A125X/A-Simple-Neural-Networl

And this project: https://github.com/dkorobchenko-nv/nn-python



So... How to use it?


First of all, you should prepare the train and test dataset. 
Each of them must be structured this way:
    [[input list for example 1], [output list for example 1]]
    [[input list for example 2], [output list for example 2]]
    and so on.

And after this shuffle them.

Secondly, provide hyperparameters to create the network.
It will be:

layers in this form:
    [amount of input neurons, hidden neurons1, ..., output neurons]

activation function:
    'tanh', 'sigmoid' or 'relu'
as a string and

type:
   'regression' or 'classification'
based on the task which you want to solve.

Next, for training you have to 
provide hyperparameters for the training, such as:

training coefficient (as a float number), 
amount of trainings 
(usually len(dataset_training) //  batch_size, positive integer), 
amount of epochs (positive integer), batch size (positive integer), 
dataset for training and dataset for testing 
which you prepared at the first step.


After training you may like to see the results. 
For this I created a couple of pretty nice methods.

calculate(input, mode='last'): 
calculating the result fron the given input. 
Can be used for the predictions, input must be the same size
(in neurons) as all the training and test input examples in the
test and training datasets.

accuracy(dataset, mode='last'):
ONLY FOR CLASSIFICATION MODELS 
providing an accuracy of prediction based on the
given dataset.

show_error(): 
showing error/accuracy during trainings.

show_determined_test(input = None, dataset_i = None):
showing correct result based on the provided dataset item.
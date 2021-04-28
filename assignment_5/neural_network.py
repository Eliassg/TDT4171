import unittest
import numpy as np
import pickle
import os
import random

class NeuralNetwork:

    def __init__(self, input_dim: int, hidden_layer: bool) -> None:
        """
        Initialize the feed-forward neural network with the given arguments.
        :param input_dim: Number of features in the dataset.
        :param hidden_layer: Whether or not to include a hidden layer.
        :return: None.
        """

        # Use the parameters below to train your feed-forward neural network.

        # Number of hidden units if hidden_layer = True.
        self.hidden_units = 25

        # Learning rate (lr).
        # This is the value of α on Line 25 in Figure 18.24.
        self.lr = 1e-3

        # Line 6 in Figure 18.24 says "repeat".
        # This is the number of times we are going to repeat. This is often known as epochs.
        self.epochs = 400

        # We are going to store the data here.
        # Since you are only asked to implement training for the feed-forward neural network,
        # only self.x_train and self.y_train need to be used. You will need to use them to implement train().
        # The self.x_test and self.y_test is used by the unit tests. Do not change anything in it.
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None

        self.input_dim = input_dim
        self.hidden_layer = hidden_layer

    def load_data(self, file_path: str = os.path.join(os.getcwd(), 'TDT4171/assignment_5/data_breast_cancer.p')) -> None:
        """
        Do not change anything in this method.

        Load data for training and testing the model.
        :param file_path: Path to the file 'data_breast_cancer.p' downloaded from Blackboard. If no arguments is given,
        the method assumes that the file is in the current working directory.

        The data have the following format.
                   (row, column)
        x: shape = (number of examples, number of features)
        y: shape = (number of examples)
        """
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            self.x_train, self.y_train = data['x_train'], data['y_train']
            self.x_test, self.y_test = data['x_test'], data['y_test']

    def train(self) -> None:
        """Run the backpropagation algorithm to train this neural network"""
        # Implement the back-propagation algorithm outlined in Figure 18.24 (page 734) in AIMA 3rd edition.
        # Only parts of the algorithm need to be implemented since we are only going for one hidden layer.
        
        # determine structure of layer in NN
        if self.hidden_layer: 
            layers = [self.input_dim, self.hidden_units, 1]
        else:
            layers = [self.input_dim, 1]
        # set weights
        self.biases =  [np.random.uniform(-0.5, 0.5, (i, 1)) for i in layers[1:]]
        self.weights = [np.random.uniform(-0.5, 0.5,(i, j)) 
                        for i, j in zip(layers[:-1], layers[1:])]
        # set targets and inputs
        inputs = self.x_train               #[(398,30)] matrix 
        target = self.y_train               #[(398,)] vector

        # Line 6 in Figure 18.24 says "repeat".
        # We are going to repeat self.epochs times as written in the __init()__ method.
        for epoch in range(self.epochs):
            if not self.hidden_layer:
                z = inputs.dot(self.weights[0]) + self.biases[0]
                activations = sigmoid(z)
                activations.resize((398,))  #predicted
                errors = target - activations

                # for each example in examples:
                for i, x_i in enumerate(inputs):
                    y_predicted = activations[i]
                    update = self.lr * (errors[i])
                    self.weights[0] += update * x_i.reshape((-1,1))
                    self.biases[0] += update * 1

            if self.hidden_layer:
                #TODO: implement generalized implementation that supports different number of hidden layers
                # for each example in examples:
                for i, x_i in enumerate(inputs):

                    z_h = inputs.dot(self.weights[0]) + self.biases[0].reshape((-1,))
                    activations_h = sigmoid(z_h)                                    # activations for hidden layer
                    z_o = activations_h.dot(self.weights[1]) + self.biases[1]
                    activations_o = sigmoid(z_o)                                    # activations for output layer
                
                    error_out = target[i] - activations_o[i]
                    delta_out = sigmoid_prime(activations_o[i]) * error_out

                    hidden_weights = self.weights[1].reshape(-1,)
                    delta_hidden = sigmoid_prime(activations_h[i]) * hidden_weights * delta_out

                    update = self.lr * delta_hidden
                    self.weights[0] += update * x_i.reshape((-1,1))                 # update weights from input layer to hidden layer (30x25) weights
                    
                    update = update.reshape((-1,1))                                 # update bias from input layer to hidden layer (1x25) bias weights
                    self.biases[0] += update

                    update = self.lr * delta_out
                    self.weights[1] += update * activations_h[i].reshape((-1,1))    # update weights from hidden layer to output layer (25x1) weights
                    
                    update = update.reshape((-1,1))                                 # update bias from hidden layer to output layer (1x1) bias weight
                    self.biases[1] += update
        pass


    def predict(self, x: np.ndarray) -> float:
        """
        Given an example x we want to predict its class probability.
        For example, for the breast cancer dataset we want to get the probability for cancer given the example x.
        :param x: A single example (vector) with shape = (number of features)
        :return: A float specifying probability which is bounded [0, 1].
        """
        #supports one hidden layer
        if self.hidden_layer:
            output_hl = sigmoid(x.dot(self.weights[0]) + self.biases[0].reshape((self.hidden_units,)))
            output = sigmoid(output_hl.dot(self.weights[1]) + self.biases[1])
            if output < 0.5:
                return 0
            else:
                return 1
        else:
            output = sigmoid(x.dot(self.weights[0]) + self.biases[0])
            if output < 0.5:
                return 0
            else:
                return 1
            
# g(z)
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
# g'(z)
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


class TestAssignment5(unittest.TestCase):
    """
    Do not change anything in this test class.

    --- PLEASE READ ---
    Run the unit tests to test the correctness of your implementation.
    This unit test is provided for you to check whether this delivery adheres to the assignment instructions
    and whether the implementation is likely correct or not.
    If the unit tests fail, then the assignment is not correctly implemented.
    """

    def setUp(self) -> None:
        self.threshold = 0.8
        self.nn_class = NeuralNetwork
        self.n_features = 30

    def get_accuracy(self) -> float:
        """Calculate classification accuracy on the test dataset."""
        self.network.load_data()
        self.network.train()

        n = len(self.network.y_test)
        correct = 0
        for i in range(n):
            # Predict by running forward pass through the neural network
            pred = self.network.predict(self.network.x_test[i])
            # Sanity check of the prediction
            assert 0 <= pred <= 1, 'The prediction needs to be in [0, 1] range.'
            # Check if right class is predicted
            correct += self.network.y_test[i] == round(float(pred))
        return round(correct / n, 3)

    def test_perceptron(self) -> None:
        """Run this method to see if Part 1 is implemented correctly."""

        self.network = self.nn_class(self.n_features, False)
        accuracy = self.get_accuracy()
        print("\n Accuracy for perceptron: " + str(accuracy))
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')

    def test_one_hidden(self) -> None:
        """Run this method to see if Part 2 is implemented correctly."""

        self.network = self.nn_class(self.n_features, True)
        accuracy = self.get_accuracy()
        print("\n Accuracy for one hidden layer: " + str(accuracy))
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')


if __name__ == '__main__':
  
    #n = NeuralNetwork(30, True)
    #n.load_data()
    #n.train()

    unittest.main()
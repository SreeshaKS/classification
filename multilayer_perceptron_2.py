import numpy as np
import random
from utils import identity, sigmoid, tanh, relu, softmax, cross_entropy, one_hot_encoding


class MultilayerPerceptron:
    def __init__(self, n_hidden=16, hidden_activation='sigmoid', n_iterations=1000, learning_rate=0.01):
        # Create a dictionary linking the hidden_activation strings to the functions defined in utils.py
        self.activation_functions = {'identity': identity, 'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu}

        # Check if the provided arguments are valid
        if not isinstance(n_hidden, int) \
                or hidden_activation not in self.activation_functions \
                or not isinstance(n_iterations, int) \
                or not isinstance(learning_rate, float):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the MultilayerPerceptron model object
        self.n_input = 0
        self.n_out = 0
        self.n_hidden = n_hidden
        self.hidden_activation = self.activation_functions[hidden_activation]
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self._output_activation = softmax
        self._loss_function = cross_entropy
        self._loss_history = []
        self._X = None
        self._y = None
        self._h_weights = None
        self._h_bias = None
        self.bias_hidden_value = -1
        self._o_weights = None
        self._o_bias = None
        self.bias_output_value = -1

    def __initialize_weights(self, x, y):
        return [[random.random() for i in range(x)] for j in range(y)]

    def _initialize(self, X, y):
        """
        Function called at the beginning of fit(X, y) that performs one hot encoding for the target class values and
        initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        np.random.seed(42)
        self.n_classes = max(y) + 1
        self._X = X
        self._y = one_hot_encoding(y)
        
        n_samples, n_features = self._X.shape
        _, n_outputs = self._y.shape

        # Hidden layer
        self._h_weights = self.__initialize_weights(n_features, self.n_hidden)
        self._h_bias = np.zeros((1, self.n_hidden))
        
        # Output layer
        self._o_weights = self.__initialize_weights(self.n_hidden, n_outputs)
        self._o_bias = np.zeros((1, n_outputs))


    # def show_err_graphic(self, v_erro, v_epoca):
    #     plt.figure(figsize=(9, 4))
    #     plt.plot(v_epoca, v_erro, "m-", color="b", marker=11)
    #     plt.xlabel("Number of Epochs")
    #     plt.ylabel("Squared error (MSE) ");
    #     plt.title("Error Minimization")
    #     plt.show()

    def predict(self, X):

        h_input = X.dot(self._h_weights) + self._h_bias
        h_output = self.hidden_activation(h_input)
        o_input = h_output.dot(self._o_weights) + self._o_bias

        return self._output_activation(o_input)

    def fit(self, X, y):
        self._initialize(X, y)

        for _ in range(self.n_iterations):
            # Forward propagation

            # input layer->hidden layer
            h_input = self._X.dot(self._h_weights) + self._h_bias
            h_output = self.hidden_activation(h_input)
            # import pdb;
            # pdb.set_trace()
            # hidden layer->output layer
            o_input = h_output.dot(self._o_weights) + self._o_bias
            prediction = self._output_activation(o_input)

            # Backpropagation

            # output layer->hidden layer
            loss_gradient_o_h = self._loss_function(self._y, prediction) \
                                   * self._output_activation(o_input, derivative=True)
            grad_v = h_output.T.dot(loss_gradient_o_h)
            grad_v0 = np.sum(loss_gradient_o_h, axis=0, keepdims=True)
            # hidden layer->input layer
            gradient_hidden_input = loss_gradient_o_h.dot(self._o_weights.T) \
                                    * self.hidden_activation(h_input, derivative=True)
            grad_w = self._X.T.dot(gradient_hidden_input)
            grad_w0 = np.sum(gradient_hidden_input, axis=0, keepdims=True)

            # Nudge weights (by gradient descent)
            self._o_weights -= self.learning_rate * grad_v
            self._o_bias -= self.learning_rate * grad_v0
            self._h_weights -= self.learning_rate * grad_w
            self._h_bias -= self.learning_rate * grad_w0

import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network
        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.model = [
            FullyConnectedLayer(n_input=n_input, n_output=hidden_layer_size),
            ReLULayer(),
            FullyConnectedLayer(n_input=hidden_layer_size, n_output=n_output),
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples
        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        
        # forward
        X_ = X.copy()
        for layer in self.model:
            X_ = layer.forward(X_)
        loss, d_pred = softmax_with_cross_entropy(X_, y)

        # backward
        for layer in reversed(self.model):
            d_pred = layer.backward(d_pred)
            d_l2 = 0
            for params in layer.params():
                param = layer.params()[params]
                l2_loss, d_reg = l2_regularization(param.value, self.reg)
                param.grad += d_reg
                loss += l2_loss
                   
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set
        Arguments:
          X, np array (test_samples, num_features)
        Returns:
          y_pred, np.array of int (test_samples)
        """
       
        pred = X
        for layer in self.model:
            pred = layer.forward(pred)

        return pred.argmax(axis=1)

    def params(self):
        result = {}
        for layer_num in range(len(self.model)):
            for i in self.model[layer_num].params():
                result[str(layer_num) + "_" + i] = self.model[layer_num].params()[i]
        return result

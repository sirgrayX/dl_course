import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, softmax, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        
        self.width, self.height, self.channels = input_shape
        self.n_classes = n_output_classes
        self.model = [
            ConvolutionalLayer(self.channels, conv1_channels, 3, 1),
            ReLULayer(),
            MaxPoolingLayer(4, 4),
            ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1),
            ReLULayer(),
            MaxPoolingLayer(4, 4),
            Flattener(),
            FullyConnectedLayer(4 * conv2_channels, self.n_classes)
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        
        for param in self.params().values():
            param.grad = np.zeros_like(param.value)
            
        # forward
        X_ = X.copy()
        for layer in self.model:
            X_ = layer.forward(X_)   
        loss, grad = softmax_with_cross_entropy(X_, y)
        
        # backward
        for layer in reversed(self.model):
            grad = layer.backward(grad)
        
        return loss

    def predict(self, X):
        X_ = X.copy()
        for layer in self.model:
            X_ = layer.forward(X_)   
        
        preds = softmax(X_)
        return np.argmax(preds, axis=1)

    def params(self):
        result = {}
        for layer_num in range(len(self.model)):
            for i in self.model[layer_num].params():
                result[str(layer_num) + "_" + i] = self.model[layer_num].params()[i]
        return result

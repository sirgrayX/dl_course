import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength * np.sum(W * W)
    grad = reg_strength * 2 * W

    return loss, grad

def softmax(predictions):
    '''
    Computes probabilities from scores
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops

    if predictions.ndim > 1:
        predictions -= np.max(predictions, axis=1).transpose().reshape((predictions.shape[0], 1))
        result = np.exp(predictions) / np.sum(np.exp(predictions), axis=1).transpose().reshape((predictions.shape[0], 1))
    else:
        predictions -= np.max(predictions)
        result = np.exp(predictions) / np.sum(np.exp(predictions))

    return result

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss
    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops

    if hasattr(target_index, '__len__'):
        ans = -np.sum(np.log(probs[np.arange(len(target_index)), target_index.reshape(1, -1)]))
    else:
        ans = -np.log(probs[target_index])

    return ans


def softmax_with_cross_entropy(preds, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    loss = cross_entropy_loss(softmax(preds.copy()), target_index)
    mask = np.zeros(preds.shape)
    if hasattr(target_index, '__len__'):
        mask[np.arange(len(target_index)), target_index.reshape(1, -1)] = 1
    else:
        mask[target_index] = 1

    d_preds = softmax(preds.copy()) - mask
    return loss, d_preds


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.cashe = X
        return np.maximum(0, X)

    def backward(self, d_out):
        """
        Backward pass
        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output
        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = d_out.copy()
        d_result[self.cashe < 0] = 0
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}
    
    def reset_grad(self):
        pass

    
    
class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X.copy()
        return X.dot(self.W.value) + self.B.value
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        raise Exception("Not implemented!")

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B
        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output
        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        self.W.grad = self.X.transpose().dot(d_out)
        E = np.ones(shape=(1, self.X.shape[0]))
        self.B.grad = E.dot(d_out)

        return d_out.dot(self.W.value.transpose())

    def params(self):
        return {'W': self.W, 'B': self.B}
    
    def reset_grad(self):
        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)

        
        
    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape
        out_height = height - self.filter_size + 2*self.padding + 1
        out_width = width - self.filter_size + 2*self.padding + 1
        output = np.zeros((batch_size, out_height, out_width, self.out_channels))
        
        X_padded = np.zeros((batch_size, height + 2 * self.padding, width + 2 * self.padding, self.in_channels))
        X_padded[:, self.padding:self.padding + height, self.padding:self.padding + width, :] = X
        self.X_cache = (X, X_padded)
        X_padded = X_padded[:, :, :, :, None]
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                X_sliced = X_padded[:, y:y+self.filter_size, x:x+self.filter_size, :]
                output[:, y, x, :] = np.sum(X_sliced * self.W.value, axis=(1,2,3)) + self.B.value
        return output

    def backward(self, d_out):   
        X, X_padded = self.X_cache
        
        batch_size, height, width, channels = X.shape
        _, out_height, out_width, out_channels = d_out.shape
        
        X_grad = np.zeros_like(X_padded)
        for y in range(out_height):
            for x in range(out_width):
                X_slice = X_padded[:, y:y + self.filter_size, x:x + self.filter_size, :, np.newaxis]
                grad = d_out[:, y, x, None, None, None, :]
                self.W.grad += np.sum(grad * X_slice, axis=0)
                X_grad[:, y:y + self.filter_size, x:x + self.filter_size, :] += np.sum(self.W.value * grad, axis=-1)
        self.B.grad += np.sum(d_out, axis=(0, 1, 2))
        return X_grad[:, self.padding:self.padding + height, self.padding:self.padding + width, :]


    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        self.X = X.copy()
        batch_size, height, width, channels = self.X.shape
        out_height = int((height-self.pool_size) / self.stride) + 1
        out_width = int((width-self.pool_size) / self.stride) + 1
        output = np.zeros((batch_size, out_height, out_width, channels))
        
        for y in range(out_height):
            for x in range(out_width):
                X_sliced = X[:, y:y+self.pool_size, x:x+self.pool_size, :]
                output[:, y, x, :] = np.max(X_sliced, axis=(1,2))
        return output
        

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        out_height = int((height-self.pool_size) / self.stride) + 1
        out_width = int((width - self.pool_size) / self.stride) + 1
        output = np.zeros(self.X.shape)
        
        for y in range(out_height):
            for x in range(out_width):
                X_sliced = self.X[:, y:y+self.pool_size, x:x+self.pool_size, :]
                grad = d_out[:, y, x, :][:, None, None, :]
                mask = (X_sliced == np.amax(X_sliced, (1, 2))[:, None, None, :])
                output[:, y:y + self.pool_size, x:x + self.pool_size, :] += grad*mask
        return output
        

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
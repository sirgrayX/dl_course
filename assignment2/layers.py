import numpy as np



def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient
    Arguments:
      W, np array - weights
      reg_strength - float value
    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
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
    """
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
    """
    # TODO: Copy from the previous assignment
    loss = cross_entropy_loss(softmax(preds.copy()), target_index)
    mask = np.zeros(preds.shape)
    if hasattr(target_index, '__len__'):
        mask[np.arange(len(target_index)), target_index.reshape(1, -1)] = 1
    else:
        mask[target_index] = 1

    d_preds = softmax(preds.copy()) - mask

    return loss, d_preds


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

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
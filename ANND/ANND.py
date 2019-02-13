import numpy as np


class Network():
    def __init__(self, layers, batch=64, learningRate=0.01, costFunc=None, dataSplit=[75, 15, 10], thresh=0.01, weights=None):
        self.layers = layers
        self.learningRate = learningRate

        if costFunc is None:
            self.costFunc = self.squarecost

        if weights is None:
            self.weights = self._initWeights()

    def _initWeights(self):
        pass

    def forwardProp(self, nparray, expectedOutputs):
        """
        Runs forward-prop on a batch.
        If batch size is 'k' and number if input vars are 'n'
        the shape of nparray is (k, n, 1) where (n, 1) is
        the column vector obtained"""

        # Starting from layers[1] (layers[0] being the input layer),
        # sequentially get the activations and store it in the
        # corresponding layer. This allows for the next layer to
        # fetch the activations (from the one before it)

        # Call the first layer
        self.layers[0].fp(nparray)

        # Sequentially calculate the following layers
        # NOTE: weight[i] is the weight matrix b/w
        # layer i, i-1
        for i in range(1, len(self.layers)):
            self.layers[i].fp(self.weights[i] @ self.layers[i-1]._activations)

        # Now, the last layer has the activations stored. Calculate
        # the error.
        return self._getError(expectedOutputs)

    def squarecost(self, lastLayerOutput, expectedOutput):
        """ Cost is sum over ( Expected - obtained )**2 """
        return np.sum(np.power(expectedOutput - lastLayerOutput, 2))


class Layer():
    class InputLayer():
        """
        Duplicates the inputs to the subsequent network"""
        def __init__(self, noOfNodes):
            self.noOfNodes = noOfNodes
            self._activations = np.zeros((noOfNodes, 1), dtype=np.float64)

        def fp(self, inputVector):
            """Forward pass"""
            self._activations = inputVector
            return inputVector

    class HiddenLayer():
        def __init__(self, noOfNodes, activFunc, bias=True, learningRate=0.01):
            self.noOfNodes = noOfNodes
            self.activFunc = activFunc
            self.lr = learningRate
            self.bias = np.ones((self.noOfNodes, 1), dtype=np.float64)
            self._activations = np.zeros((self.noOfNodes, 1), dtype=np.float64)

        def fp(self, nparray):
            """
            Forward Pass:
            Returns the activations of the layer, given inputs
            Input: nx1 numpy array (W*A) [from prev. layer]
            Output: ActivFunc(W*A + B)
            Where W are the weights between this and the prev.
            layer, A are the activations of the prev. layer,
            B is the bias of this layer"""
            self._activations = self.activFunc(nparray + self.bias)
            return self._activations

        def derivative(self):
            pass


class Funcs():
    # Class of functions that can be added to
    # layers
    # Each class is a function, that wraps its
    # properties: the function itself and
    # its derivative (.d method)
    class RELU():
        @staticmethod
        def forward(nparray):
            return (nparray > 0) * nparray

        @staticmethod
        def d(ndarrar):
            return (ndarrar > 0) * 1

    class Sigmoid():
        @staticmethod
        def forward(nparray):
            return np.exp(nparray)/(1 + np.exp(nparray))

        @staticmethod
        def d(nparray):
            return Funcs.Sigmoid.forward(nparray) * (1 - Funcs.Sigmoid.forward(nparray))

    class Tanh():
        @staticmethod
        def forward(nparray):
            return np.tanh(nparray)

        @staticmethod
        def d(nparray):
            return 1 - np.power(np.tanh(nparray), 2)

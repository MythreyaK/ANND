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

    def forwardProp(self, nparray):
        pass

    def squarecost(self, lastLayerOutput, expectedOutput):
        """ Cost is sum over ( Expected - obtained )**2 """
        return np.sum(np.power(expectedOutput - lastLayerOutput, 2))


class Layer():
    class InputLayer():
        """
        Duplicates the inputs to the subsequent network"""
        @staticmethod
        def fp(inputVector):
            """Forward pass"""
            return inputVector

    class HiddenLayer():
        def __init__(self, noOfNodes, activFunc, bias=True, learningRate=0.01):
            self.noOfNodes = noOfNodes
            self.activFunc = activFunc
            self.lr = learningRate
            self.bias = np.ones((self.noOfNodes, 1), dtype=np.float64)

        def fp(self, nparray):
            """
            Forward Pass:
            Returns the activations of the layer, given inputs
            Input: nx1 numpy array (W*A) [from prev. layer]
            Output: ActivFunc(W*A + B)
            Where W are the weights between this and the prev.
            layer, A are the activations of the prev. layer,
            B is the bias of this layer"""
            return self.activFunc(nparray + self.bias)

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

import numpy as np


class Network():
    def __init__(self, inAndOut, batch=64, learningRate=0.01, costFunc=None, dataSplit=[75, 15, 10], thresh=0.01):
        self.learningRate = learningRate
        self.batchSize = batch
        self.weights = []
        self.layers = []
        self.inAndOut = inAndOut
        if costFunc is None:
            self.costFunc = self.squarecost

    def Sequential(self, *layers):
        """
        Takes a series of Layers (of type 'Layer')
        to create those Hidden layers. Input layer is
        added automatically. Output is the last layer."""

        # Add the input layer
        self.layers.append(Layer._InputLayer(self.inAndOut[0]))

        # Iterate over the *layers and add them
        # to self.layers
        for i in range(len(layers)):
            layers[i].lr = self.learningRate
            self.layers.append(layers)

        # Sanity check of the shape of last layer
        if self.layers[-1].noOfNodes != self.inAndOut[1]:
            raise ValueError("Output layer's nodes defined (" + str(self.inAndOut[1]) +
                             ") vs given (" + self.layers[-1].noOfNodes + ") mismatch")

        # Now that the layers are in place, initialize
        # the weights
        self.weights = self._initWeights()

    def _initWeights(self):
        """
        Initialize weights from a random-normal distribution
        and scale them down by the number of nodes in the preceding
        layer. This reduces the chances that the combined
        effects of their linear combination is large"""

        # Iterate over the weights and init weights
        # NOTE: weight[i] is the weight matrix b/w
        # layer i, i-1. This notation implies weights[0] is
        # not used
        weights = [0]
        for i in range(1, len(self.layers)):
            arr = np.random.randn(self.layers[i].noOfNodes,
                                  self.layers[i-1].noOfNodes) \
                / np.sqrt(self.layers[i-1].noOfNodes)

            arr = arr.astype(np.float64)
            weights.append(arr)

        return weights

    @property.getter
    def weightsL2Norm(self):
        """
        The L2 norm of the weights is used for regularization
        to try and keep the abs(weights) in control"""
        ret = 0
        for i in range(self.weights):
            ret += np.sum(i**2)
        return ret**0.5

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
        # layer i, i-1. This notation implies weights[0] is
        # not used
        for i in range(1, len(self.layers)):
            self.layers[i].fp(self.weights[i] @ self.layers[i-1]._activations)

        # Now, the last layer has the activations stored. Return
        # the last layer's activations
        return self.layers[-1]._activations

    def backProp(self, errors):
        """
        Use the errors to calculate the gradient of the
        cost w.r.t the weights, biases and update them
        in an effort to make the Network 'learn'"""
        pass

    def squarecost(self, lastLayerOutput, expectedOutput):
        """ Cost is sum over ( Expected - obtained )**2 """
        return np.sum(np.power(expectedOutput - lastLayerOutput, 2))


class Layer():
    class _InputLayer():
        """
        Duplicates the inputs to the subsequent network"""

        def __init__(self, noOfNodes):
            self.noOfNodes = noOfNodes
            self._activations = None
            # Layer._z = W.A + B
            self._z = None

        def fp(self, inputVector):
            """Forward pass"""
            self._activations = inputVector
            self._z = inputVector
            return inputVector

    class HiddenLayer():
        def __init__(self, noOfNodes, activFunc):
            self.noOfNodes = noOfNodes
            self.activFunc = activFunc
            self.lr = None
            self.bias = np.zeros((self.noOfNodes, 1), dtype=np.float64)
            self._activations = None
            self._z = None

        def fp(self, nparray):
            """
            Forward Pass:
            Returns the activations of the layer, given inputs
            Input: nx1 numpy array (W*A) [from prev. layer]
            Output: ActivFunc(W*A + B)
            Where W are the weights between this and the prev.
            layer, A are the activations of the prev. layer,
            B is the bias of this layer"""
            self._z = nparray + self.bias
            self._activations = self.activFunc(self._z)
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

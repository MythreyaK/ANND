import numpy as np
import matplotlib.pyplot as plt


class Network():
    def __init__(self, inpNodes, outNodes, dataArray, costFunc,
                 optimizer=None, batch=64, learningRate=0.01, dataSplit=[75, 15, 10]):
        self.learningRate = learningRate
        self.batchSize = batch
        self.weights = []
        self.layers = []
        self.inAndOut = [inpNodes, outNodes]
        self.costFunc = costFunc
        # A way to store the errors
        # after each forward prop
        # Store the batch indices
        self.__splitFr = dataSplit
        self.__trainSet = None
        self.__valSet = None
        self.__testSet = None
        if type(dataArray) is np.ndarray:
            self.__dataSet = dataArray
        else:
            raise ValueError("Only numpy arrays are supported. Ensure last\
                    column of data has the classes/expected outputs")
        self._initPlot()

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
            self.layers.append(layers[i])

        # Sanity check of the shape of last layer
        if self.layers[-1].noOfNodes != self.inAndOut[1]:
            raise ValueError("Output layer's nodes defined (" + str(self.inAndOut[1]) +
                             ") vs given (" + self.layers[-1].noOfNodes + ") mismatch")

        # Now that the layers are in place, initialize
        # the weights
        self.weights = self._initWeights()
        self.__dataSet = self.__cleanAndNormalize(self.__dataSet)

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

    @property
    def weightsL2Norm(self):
        """
        The L2 norm of the weights is used for regularization
        to try and keep the abs(weights) in control"""
        ret = 0
        for i in range(self.weights):
            ret += np.sum(i**2)
        return ret**0.5

    def forwardProp(self, nparray):
        """
        Runs forward-prop on a batch.
        If batch size is 'k', number of input vars are 'n',
        the shape of nparray is (k, n, 1) where (n, 1) is
        the column vector representing an input"""

        # Starting from layers[1] (layers[0] being the input layer),
        # sequentially get the activations and store it in the
        # corresponding layer. This allows for the next layer to
        # fetch the activations (from the one before it)

        # Call the first layer
        self.layers[0](nparray)

        # Sequentially calculate the following layers
        # NOTE: weight[i] is the weight matrix b/w
        # layer i, i-1. This notation implies weights[0] is
        # not used
        for i in range(1, len(self.layers)):
            self.layers[i](self.weights[i] @ self.layers[i-1]._activations)

        # Now, the last layer has the activations stored. Return
        # the last layer's activations
        return self.layers[-1]._activations

    def backProp(self, errors):
        """
        Use the errors to calculate the gradient of the
        cost w.r.t the weights, biases and update them
        in an effort to make the Network 'learn'"""

        # dw, db list so that all are updated at once. This is
        # to also facilitate regularization (which uses the old
        # weights, before updates)
        dws, dbs = [0], [0]
        # Last layer, special case, dC/dA is (act - pred)
        loGrad = - errors

        for i in range(len(self.weights) - 1, 0, -1):
            loGrad = loGrad * self.layers[i]._fderiv()
            avgAct = np.average(self.layers[i-1]._activations, axis=0)
            dws.insert(1, -loGrad @ avgAct.T)
            dbs.insert(1, -loGrad)
            loGrad = self.weights[i].T @ loGrad

        # Call the function that updates the weights and biases
        # based on chosen update method
        self.__updateParameters(dws, dbs)

    def __updateParameters(self, delWs, delBs):
        for i in range(len(self.weights) - 1, 0, -1):
            self.weights[i] -= self.layers[i].lr * delWs[i]
            self.layers[i].bias -= self.layers[i].lr * delBs[i]

    def Train(self, noOfEpochs):
        """
        Run the training, validation loop with the
        parameters defined"""

        # Create train-test-validation data split
        self.__splitData()

        trainErrors = []
        valErrors = []

        for epoch in range(noOfEpochs):
            # Create a list of indices from 0 - len(trainData)
            # and create batchs of size self.batchSize and pass
            # it to the forward pass.

            # Pass the batches, get the error, backProp.
            # Rinse and repeat. Oh and store the errors for
            # plotting purposes
            avgEr = 0
            for batchNum, batch, expectedVector in self.__getBatch():
                self.forwardProp(batch)
                # Last layer's activations are stored by
                # the forward prop function, so no need to
                # pass as parameter
                errors = self.__getErrors(expectedVector)
                avgEr += np.sum(errors**2)
                self.backProp(errors)

            # Run validation
            self.forwardProp(self.__customReshape(
                self.__valSet[:, :-1], diffDim=len(self.__valSet))
            )
            valErrors.append(np.sum(
                self.__getErrors(
                    self.__getExpectedOutput(
                        self.__valSet[:, -1], diffDim=len(self.__valSet))
                )**2)
            )
            trainErrors.append(avgEr)
            self._updatePlot(trainErrors, valErrors)
        self._keepPlot()

    def __splitData(self):
        """
        Splits data in the ratio specified by the user."""

        # Generate a list of random indices that are used to
        # pick data-points

        randInx = np.random.permutation(len(self.__dataSet))
        trainFr = int(np.ceil(self.__splitFr[0] * len(randInx))/100)
        valFr = int(np.ceil(self.__splitFr[1] * len(randInx)/100) + trainFr)

        self.__trainSet = self.__dataSet[randInx[:trainFr]]
        self.__valSet = self.__dataSet[randInx[trainFr:valFr]]
        self.__testSet = self.__dataSet[randInx[valFr:]]

    def __getBatch(self):
        """
        Create batches out of the dataset with each batch
        containing self.batchSize data points"""

        # If batch size does not divide the train set, repeat points
        # instead of dropping them
        ts = len(self.__trainSet)
        bs = self.batchSize
        # Say batch size is 21, trainSetlength is 50.
        # 50 % 21 is 8. So to get length that can give 0-remainder,
        # we do 50 - (50%21) + 21; implies we have numbers from range
        # 0 to 63. These indices are used to get the elements from
        # the train set. But note, indices 50 to 62 are invalid! So,
        # just do modulo 50, making all the indices lie in range
        # 0 to 50, but a few indices repeat (meaning we are reusing
        # data points, which is exactly what we want)
        permuteNumberRange = ts + bs*(ts % bs != 0) - (ts % bs)

        shufflInx = np.random.permutation(permuteNumberRange) % ts
        numOfBatches = len(shufflInx)//self.batchSize

        # Last columns stores the expected values, so
        # remove them (after saving them)
        expVals = self.__trainSet[:, -1]
        trainSet = self.__trainSet[:, :-1]

        for i in range(numOfBatches - 1):
            # Get the data points from the train set based on
            # the indices stored in shufflInx, but only a few
            # at a time (this 'few' is same as the batch size)
            # and reshape them to (batchSize, noOfCols in data,
            # 1). The -1 in reshape instructs numpy to infer the
            # missing shape parameter
            bs = self.batchSize
            batch = self.__customReshape(
                trainSet[shufflInx[i * bs:(i + 1)*bs]]
            )

            # getClassifier:
            # Assume a data point that belongs to class 1* (out
            # of 4 possible classes, this means out network
            # has 4 output nodes), but our network predicts
            # that it belongs to class 3. The output is a
            # column vector, hence to calculat the error,
            # the 'expected' array needs to be a vector too.
            # So, the output is something like
            # [~=0, ~=1, ~=0, ~=0];
            # by using the '3' from the data, we construct
            # the expected vector as, [0, 0, 0, 1]
            # *= Classes are assumed to be 0-indexed in this
            # example
            yield i, batch, self.__getExpectedOutput(
                expVals[shufflInx[i*bs: (i+1)*bs]]
            )

    def __customReshape(self, ndarray, diffDim=None):
        if diffDim is None:
            return ndarray.reshape(self.batchSize, -1, 1)
        else:
            return ndarray.reshape(diffDim, -1, 1)

    def __getClassifier(self, inx, shp):
        arr = np.zeros((shp, self.inAndOut[1], 1))
        arr[[x for x in range(len(arr))], inx.astype(np.int32)] = 1
        return arr

    def __getErrors(self, expectedVals):
        return np.average(
            (expectedVals - self.layers[-1]._activations),
            axis=0
        )

    def _initPlot(self):
        plt.show()
        self.axes = plt.gca()
        self._trEplt, self._valEplt = \
            self.axes.plot([], [], 'b-', [], [], 'g-')

    def _updatePlot(self, tr, val):
        self._trEplt.set_data([t for t in range(len(tr))], tr)
        self._valEplt.set_data([t for t in range(len(val))], val)
        self.axes.set_xlim(0, len(tr))
        self.axes.set_ylim(-0.25, max(tr) + max(tr)*0.10)
        plt.pause(1e-17)

    def _keepPlot(self):
        plt.show()


    def __cleanAndNormalize(self, ndarray):
        data = np.nan_to_num(ndarray).astype(np.float64)
        # If porblem type is classification (implies more than one output
        # node), don't normalize the last column, as it contins the class labels
        # If problem type is prediction, (implies only one output node),
        # then normalize the last column to 0 to 0.9 if sigmoid,
        # else -0.9 to 0.9
        if self.layers[-1].noOfNodes > 1:
            # Classification problem. Normalize all but the last
            # column
            tmp = self.__normalize(data[:, :-1], -0.9, 0.9)
            lastCol = data[:, -1].reshape(len(data), 1)
            return np.concatenate((tmp, lastCol), axis=1)
        else:
            # Prediction problem. Scale cols to +-0.9 if all but Sigmoid
            # Also store the min and max of data, so that de-normalization
            # can be done
            self._dmin = min(ndarray[:, -1])
            self._dmax = max(ndarray[:, -1])
            if isinstance(self.layers[-1].activFunc, Funcs.Sigmoid):
                tmp = self.__normalize(data[:, :-1], -0.9, 0.9)
                lastCol = self.__normalize(
                    data[:, -1], 0, 0.9).reshape(len(data), 1)
                return np.concatenate((tmp, lastCol), axis=1)
            else:
                return self.__normalize(data, -0.9, 0.9)

    def __normalize(self, ndarray, low, high):
        colsRange = np.max(ndarray, axis=0) - np.min(ndarray, axis=0)
        tmp = low + (high-low)*(ndarray - np.min(ndarray, axis=0))/colsRange
        return tmp

    def __deNorm(self, ndarray):
        # Returns a value ranging from the least to the
        # max in the dataset's last column
        # If last output is Sigmoid, then scale
        # (0, 1) to (min of dataset to max of dataset)
        # [last col only]
        dMax = self._dmax
        dMin = self._dmin
        print(dMax, dMin)
        if isinstance(self.layers[-1].activFunc, Funcs.Sigmoid):
            # Scale (0, 1) -> (dmin, dmax)
            return dMin + ndarray*(dMax - dMin)
        else:
            # scale (-1, 1) -> (dmin, dmax)
            return dMin + 0.5*(ndarray + 1) * (dMax - dMin)

    def __getExpectedOutput(self, ndarray, diffDim=None):
        # Again, if classification, call getClassifier
        # else jsut return the array, since we've already
        # normalized in the _cleanAndNormalize function
        if diffDim is None:
            shp = self.batchSize
        else:
            shp = diffDim

        if self.layers[-1].noOfNodes == 1:
            return ndarray.reshape(shp, 1, 1)
        else:
            return self.__getClassifier(ndarray, shp)

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

        def __call__(self, inputVector):
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

        def __call__(self, nparray):
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

        def _fderiv(self):
            """
            Returns the derivative of the activation
            function at the Z values"""
            return np.average(self.activFunc.d(self._z), axis=0)


class Funcs():
    # Class of functions that can be added to
    # layers
    # Each class is a function, that wraps its
    # properties: the function itself and
    # its derivative (.d method)
    class RELU():
        @staticmethod
        def __call__(nparray):
            return (nparray > 0) * nparray

        @staticmethod
        def d(ndarrar):
            return (ndarrar > 0) * 1

    class Sigmoid():
        @staticmethod
        def __call__(nparray):
            temp = nparray
            temp[temp > 100] = 100
            temp[temp < -100] = -100
            return np.exp(temp)/(1 + np.exp(temp))

        @staticmethod
        def d(nparray):
            p = Funcs.Sigmoid()
            return p(nparray) * (1 - p(nparray))

    class Tanh():
        @staticmethod
        def __call__(nparray):
            return np.tanh(nparray)

        @staticmethod
        def d(nparray):
            return 1 - np.power(np.tanh(nparray), 2)

import numpy as np


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

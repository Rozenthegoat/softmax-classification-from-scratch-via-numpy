import numpy as np
import math


def Sigmoid(w, X):

    input = -w.T*X
    sigmoid = 1 / (1 + math.exp(input))

    return sigmoid


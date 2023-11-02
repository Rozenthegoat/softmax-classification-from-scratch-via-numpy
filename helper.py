import numpy as np
from math import exp, log

num_of_class = 10 # number of classes

def Softmax(predict):
    """
    predict     : result of w*X, ten predicted values, 1*10
    """
    softmax = np.zeros(predict.shape)
    exp_all = 0
    for i in range(predict.shape[0]):
        y_i = predict[i]
        exp_all += exp(y_i)
    for i in range(predict.shape[0]):
        p_i = exp(predict[i]) / exp_all
        softmax[i] = p_i
    return softmax

def CrossEntropyLoss(p_softmax, y_true):
    """
    p_softmax   : 10*1 metrix, p_0 to p_9
    y_true      : ground truth value of current y
    """
    one_hot = get_one_hot(y_true)
    loss = 0
    for i in range(p_softmax.shape[0]):
        p_i = p_softmax[i]
        y_hat = one_hot[i] # aka y_hat
        # print(y_true, log(p_i))
        loss += - y_hat * log(p_i)
    return loss

def get_one_hot(y_true):
    one_hot = [0] * num_of_class
    one_hot[y_true] = 1
    return one_hot

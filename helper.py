import numpy as np
from math import exp, log


def Softmax(y_k, ground_truth):
    """
    y_k         : aka y_pred
    ground_truth: full set of ground_truth value
    """
    p_k = 0
    exp_all = 0
    for i in range(len(ground_truth)):
        y_i = ground_truth[i]
        exp_all += exp(y_i)
    p_k = exp(y_k) / exp_all
    return p_k

def CrossEntropyLoss(predict, ground_truth):
    """
    Calculate in terms of every single image
    predict     : 60000 length of list
    ground_truth: 60000 length of list, aka y_train
    """
    loss = 0
    if len(predict) != len(ground_truth):
        raise ValueError("雪豹闭嘴")
    for i in range(len(predict)):
        y_pred = predict[i]
        y_true = ground_truth[i]
        p_i = Softmax(y_pred, ground_truth)
        loss += - y_true * log(p_i)
    return loss

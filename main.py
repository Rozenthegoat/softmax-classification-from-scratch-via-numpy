import numpy as np
import pandas as pd
from helper import *
from helper import Sigmoid
from sklearn.model_selection import train_test_split
import mnist
# from mnist import LoadMNIST
from os.path import join
import time

class Weight():
    def __init__(self, param_num):
        """
        param_num: number of learnable parameters
                    in this case, image will be flatten to a 784 by 1 matrix,
                    each pixel will have its own learnable parameter, i.e., theta.


        self.metrix: the matrix of learnable parameters
        self.grad  : the metrix of gradients
        """
        self.metrix = np.zeros((param_num, 1))
        self.grad = np.zeros((param_num, 1))

    def update_weight(self, lr=0.001, grad=None):
        """
        After finishing iterating one batch, the parameter matrix needs to be updated. 
        lr  : learning rate, 0.001 by default.
        grad: the metrix of gradients
        """
        for i in range(self.metrix.shape[0]):
            self.metrix[i] = self.metrix[i] - lr * self.grad[i]

        return self.metrix

    def get_weight_metrix(self):
        return self.metrix


def CrossEntropyLoss(w, b, predict, ground_truth):
    loss = 0
    for i in range(predict.shape[0]):
        y_gt = ground_truth[i]
        y_pred = predict[i]
        y_sig = Sigmoid(y_gt, y_pred)

    return loss

def step_gradient_update(w, b, input, ground_truth, lr):
    grad_w = 0
    grad_b = 0
    loss = 0

    for i in range(data.shape[0]):
        x = input[i]
        y = ground_truth[i]
        pred = np.matmul(w.T, x)
        loss = CrossEntropyLoss(w, b, pred, y)
        grad_w += -2*y*x + 2*(x**2)*m + 2*x*b
        grad_b += -2*y + 2*m*x + 2*b

    new_w = w - lr * grad_w / input.shape[0]
    new_b = b - lr * grad_b / input.shape[0]

    return new_w, new_b

if __name__ == '__main__':
    # data preparation, check mnist.py carefully
    x_train, y_train, x_test, y_test = mnist.LoadMNIST()

    # Initialize learnable parameters
    param_num = x_train[0].flatten().shape[0]
    weight = Weight(param_num=param_num)

    # hyperparameters
    lr = 1e-3
    epochs = 5
    batch_size = 20

    train_start_time = time.time()
    # training procedure
    for i in range(epochs):
        for img_idx in range(len(x_train)):
            x = x_train[img_idx].flatten() # flatten all rows into one row
            ground_truth = y_train[img_idx]
            w = weight.get_weight_metrix()
            # print(w)
            # print(w.shape)
            # print(type(w))
            pred_y = np.matmul(x, w) # linear combination of 1*784 metrix and 784 * 1 metrix
            print(pred_y)
            print(pred_y.shape)
            # TODO: calculate gradient
            # TODO: gradient decent
            # TODO: calculate cross entropy loss

    train_end_time = time.time()
    print(f"Finish training, cost {train_end_time-train_start_time:.2f} sec.")

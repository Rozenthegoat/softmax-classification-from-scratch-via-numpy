import numpy as np
import pandas as pd
from helper import *
from helper import CrossEntropyLoss
from sklearn.model_selection import train_test_split
import mnist
# from mnist import LoadMNIST
from os.path import join
import time
from tqdm import tqdm

class Weight():
    def __init__(self, param_num):
        """
        param_num: number of learnable parameters
                    in this case, image will be flatten to a 784 by 1 matrix,
                    each pixel will have its own learnable parameter, i.e., theta.
        self.metrix: the matrix of learnable parameters
        self.loss  : loss
        """
        self.weight_metrix = np.zeros((param_num, 1))
        self.loss = 0

    def update_weight_metrix(self, grad, lr):
        """
        After finishing iterating one batch, the parameter matrix needs to be updated. 
        grad: the metrix of gradients
        """
        for i in range(self.weight_metrix.shape[0]):
            self.weight_metrix[i] = self.weight_metrix[i] - lr * grad[i]

        return self.weight_metrix

    def get_weight_metrix(self):
        return self.weight_metrix


class Gradient():
    def __init__(self, param_num):
        """
        param_num: number of learnable parameters
                    in this case, image will be flatten to a 784 by 1 matrix,
                    each pixel will have its own learnable parameter, i.e., theta.
        gradient_metrix: same shape as learnable parameter metrix
        """
        self.gradient_metrix = np.zeros((param_num, 1))

    def calculate_gradient(self, curr_x, curr_y, ground_truth):
        """
        curr_x       : current training sample, 1*784 image here
        curr_y       : current label, single digit from 0 to 9
        ground_truth : full set of ground truth value, y_train
        """
        y_i = curr_y
        p_j = Softmax(curr_y, ground_truth)
        for j in range(self.gradient_metrix.shape[0]):
            if j == y_i:
                self.gradient_metrix[j] = - (1 - p_j) * curr_x[j]
            else:
                self.gradient_metrix[j] = p_j * curr_x[j]

        return self.gradient_metrix


if __name__ == '__main__':
    # data preparation, check mnist.py carefully
    x_train, y_train, x_test, y_test = mnist.LoadMNIST()

    # hyperparameters
    lr = 1e-3
    epochs = 5

    # Initialize learnable parameters
    param_num = x_train[0].flatten().shape[0]
    weight = Weight(param_num=param_num)

    # Initialize learnable parameters
    gradients = Gradient(param_num=param_num)

    accuracy = 0
    train_start_time = time.time()
    # training procedure
    for i in range(epochs):
        print(f"################  Start training at epoch {i+1}  ################")
        prediction = [] # record all predicted y obtained in this current training epoch
        for img_idx in tqdm(range(len(x_train))):
            x = x_train[img_idx].flatten() # flatten all rows into one row
            ground_truth = y_train[img_idx]
            w = weight.get_weight_metrix()
            pred_y = np.matmul(x, w) # linear combination of 1*784 metrix and 784 * 1 metrix
            prediction.append(pred_y)
            # calculate gradient, aka currrent_gradient
            grad_after_this_sample = gradients.calculate_gradient(x, pred_y, y_train)
            # TODO: gradient decent
            weight.update_weight_metrix(grad_after_this_sample, lr)
        # Calculate cross entropy loss
        loss = CrossEntropyLoss(prediction, y_train)

        # testing
        correct_num = 0
        w = weight.get_weight_metrix()
        print(f"################  Start testing at epoch {i+1}  ################")
        for test_idx in tqdm(range(len(x_test))):
            x = x_test[test_idx].flatten()
            ground_truth = y_test[test_idx]
            pred_y = np.matmul(x, w)
            if pred_y == ground_truth:
                correct_num += 1
        accuracy = correct_num / len(x_test)
        print(f"###  Finish training epoch {i+1}, loss: {loss:.4f}, accuracy: {100*accuracy:.2f}%  ###")

    train_end_time = time.time()
    print(f"Finish training, final accuracy: {100*accuracy:.2f}%, cost {train_end_time-train_start_time:.2f} sec.")

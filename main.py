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
        self.weight_metrix = np.zeros((10, param_num)) # 10 classes
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
        self.gradient_metrix = np.zeros((param_num, 10))

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
                self.gradient_metrix[j] = - (1 - p_j) * curr_x
            else:
                self.gradient_metrix[j] = p_j * curr_x

        return self.gradient_metrix

    def linear_layer(X):
        """
        aka FC layer
        X: current data, 784*1 matrix
        """
        grad_y_to_w = X.T
        return grad_y_to_w

    def softmax_layer(p_softmax, grad_J_to_p):
        """
        y_pred: current predicted y value
        y_true: current true y value, a scalar, representing j
        # p_softmax: 1*10 softmax probability of current data
        ground_truth: full set of y_train, 1*10 list

        Return:
        grad_p_to_y: 1*10 numpy array
        """
        grad_J_to_y = np.zeros(p_softmax.shape)
        for i in range(grad_J_to_y.shape[0]):
            sum = 0
            for j in range(grad_J_to_y.shape[0]):
                if i == j:
                    continue
                else:
                    sum += grad_J_to_p[j] * p_softmax[j] * p_softmax[i]
            grad_J_to_y[i] = grad_J_to_p[i] * p_softmax[i] * (1 - p_softmax[i]) - sum
        return grad_J_to_y

    def loss_layer(p, y_true):
        """
        p: p_softmax, 1*10 list
        y_true: current true y
        
        Return
        grad_J_to_p: 10*1 numpy array 
        """
        # print(p, y_true)
        one_hot = get_one_hot(y_true)
        grad_J_to_p = p.copy()
        for i in range(len(grad_J_to_p)):
            y_hat = one_hot[i]
            grad_J_to_p[i] = - y_hat / p[i]
        return grad_J_to_p


if __name__ == '__main__':
    # data preparation, check mnist.py carefully
    x_train, y_train, x_test, y_test = mnist.LoadMNIST()
    ground_truth = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # hyperparameters
    lr = 1e-3
    epochs = 5
    batch_size = 100

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
            # get X and y
            x = x_train[img_idx].flatten().reshape(-1, 1) # flatten all rows, reshape to a column vector
            y_true = y_train[img_idx]
            w = weight.get_weight_metrix()
            # BEGIN Forward
            y_pred = np.matmul(w, x) # linear combination of 1*784 metrix and 784 * 10 metrix
            p_softmax = Softmax(y_pred)
            loss = CrossEntropyLoss(p_softmax, y_true)
            # END Forward

            # BEGIN Backward
            grad_J_to_p = Gradient.loss_layer(p_softmax, y_true)
            grad_J_to_y = Gradient.softmax_layer(p_softmax, grad_J_to_p)
            print(grad_J_to_y)
            grad_y_to_w = Gradient.linear_layer(x)
            check = f"""
            grad_J_to_p:{grad_J_to_p.shape}
            grad_p_to_y:{grad_J_to_y.shape}
            grad_y_to_w:{grad_y_to_w.shape}
            np.matmul(grad_J_to_p, grad_p_to_y.T): {(np.matmul(grad_J_to_p.T, grad_y_to_w)).shape}
            """
            # print(check)
            grad_J_to_w = np.matmul(grad_J_to_y.T, grad_y_to_w)
            print(grad_J_to_w.shape)
            # END Backward

            # calculate gradient, aka currrent_gradient
            grad_after_this_sample = gradients.calculate_gradient(x, pred_y, y_train)
            # TODO: gradient decent
            weight.update_weight_metrix(grad_after_this_sample, lr)
            # Calculate cross entropy loss

        # testing
        correct_num = 0
        w = weight.get_weight_metrix()
        print(f"################  Start testing at epoch {i+1}  ################")
        for test_idx in tqdm(range(int(len(x_test)/10))):
            x = x_test[test_idx].flatten()
            ground_truth = y_test[test_idx]
            pred_y = np.matmul(x, w)
            if pred_y == ground_truth:
                correct_num += 1
        accuracy = correct_num / len(x_test)
        print(f"###  Finish training epoch {i+1}, loss: {loss:.4f}, accuracy: {100*accuracy:.2f}%  ###")

    train_end_time = time.time()
    print(f"Finish training, final accuracy: {100*accuracy:.2f}%, cost {train_end_time-train_start_time:.2f} sec.")

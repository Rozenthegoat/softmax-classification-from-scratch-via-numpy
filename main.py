import numpy as np
import pandas as pd
from helper import *
from helper import Sigmoid
from sklearn.model_selection import train_test_split


def CrossEntropyLoss(predict, ground_truth):
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
        loss = CrossEntropyLoss(pred, y)
        grad_w += -2*y*x + 2*(x**2)*m + 2*x*b
        grad_b += -2*y + 2*m*x + 2*b

    new_w = w - lr * grad_w / input.shape[0]
    new_b = b - lr * grad_b / input.shape[0]

    return new_w, new_b

if __name__ == '__main__':
    # data preparation
    data = pd.read_csv('data.csv')
    data = np.array(data)
    X = data[:-1]
    y = data[-1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # hyperparameters
    lr = 1e-3
    epochs = 5
    batch_size = 20
    
    # learnable parameters
    w = np.zeros(X.shape[1])
    b = 0

    for i in range(epochs):
        w, b = step_gradient_update(w, b, x_train, y_train, lr)

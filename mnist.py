#
# This is a sample demonstration of how to read MNIST Dataset
# https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook
# 
import numpy as np
import struct
from array import array
from os.path import join
import time

# 
# MNIST Data Loader Class
#
class MnistDataLoader(object):
    def __init__(self, 
                 training_images_filepath, training_labels_filepath,
                 testing_images_filepath, testing_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.testing_images_filepath = testing_images_filepath
        self.testing_labels_filepath = testing_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
            image_data = np.array(image_data) # obtain 60000 * 28 * 28

        images = []
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #                       Original implementation from Kaggle                               #
        # Here I read the data as 60000 length list, containing (28, 28) np.array for each image  #
        #                                                                                         #
        # for i in range(size):                                                                   #
        #     images.append([0] * rows * cols)                                                    #
        # for i in range(size):                                                                   #
        #     img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])                   #
        #     img = img.reshape(28, 28)                                                           #
        #     images[i][:] = img                                                                  #
        #                                                                                         #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # 
        # My own modification
        # 
        for i in range(size):
            img = np.array(image_data[i * rows * cols : (i + 1) * rows * cols]) # get ith image pixel-wise
            img = img.reshape(28, 28)
            images.append(img)

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath,
            self.training_labels_filepath
        )
        x_test, y_test = self.read_images_labels(
            self.testing_images_filepath,
            self.testing_labels_filepath
        )
        return (x_train, y_train), (x_test, y_test)

def LoadMNIST():
    input_path = './MNIST'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    # Load MNIST dataset
    mnist_dataloader = MnistDataLoader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    curr_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"Start loading MNIST dataset, current time: {curr_time}")
    load_start_time = time.time()
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    load_end_time = time.time()
    print(f"Finish loading MNIST dataset, cost {load_end_time-load_start_time:.2f} sec.")
    mnist_description = f"""
########################################################################################
Description of MNIST Dataset

    Number of train image: {len(x_train)}
    Number of train label: {len(y_train)}
    Number of test image: {len(x_test)}
    Number of test label: {len(y_test)}

MNIST dataset consists 60000 training data, 10000 testing data.
x_train, y_train, x_test, y_test are all in type of {type(x_train)}.
For each image, its type is {type(x_train[0])}, the shape of it is: {x_train[0].shape}.
########################################################################################
    """
    print(mnist_description)

    return x_train, y_train, x_test, y_test

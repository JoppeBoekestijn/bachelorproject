from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tflearn
from tflearn.data_utils import image_preloader
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

# Global variables
img_size = 64
num_channels = 3
num_classes = 10


def cnn_net():
    # input_layer = tf.reshape(images, [-1, img_size, img_size, num_channels])
    network = tflearn.layers.core.input_data(shape=[None, img_size, img_size, num_channels], name='input')
    network = tflearn.conv_2d(network, 32, 3, activation='relu', name='conv1', regularizer='L2')
    network = tflearn.max_pool_2d(network, 2)
    network = tflearn.layers.normalization.local_response_normalization(network)
    network = tflearn.conv_2d(network, 64, 3, activation='relu', name='conv1', regularizer='L2')
    network = tflearn.max_pool_2d(network, 2)
    network = tflearn.layers.normalization.local_response_normalization(network)
    network = tflearn.layers.core.fully_connected(network, 128, activation='tanh')
    network = tflearn.layers.core.dropout(network, 0.8)
    network = tflearn.layers.core.fully_connected(network, 256, activation='tanh')
    network = tflearn.layers.core.dropout(network, 0.8)
    network = tflearn.layers.core.fully_connected(network, num_classes, activation='softmax')
    network = tflearn.layers.estimator.regression(network, optimizer='adam',
                                                  learning_rate=0.01,
                                                  loss='categorical_crossentropy',
                                                  name='target')
    return network


def alex_net():
    network = tflearn.layers.core.input_data(shape=[None, img_size, img_size, num_channels], name='input')
    network = tflearn.conv_2d(network, 96, 11, strides=4, activation='relu')
    network = tflearn.max_pool_2d(network, 3, strides=2)
    network = tflearn.layers.normalization.local_response_normalization(network)
    network = tflearn.conv_2d(network, 256, 5, activation='relu')
    network = tflearn.max_pool_2d(network, 3, strides=2)
    network = tflearn.layers.normalization.local_response_normalization(network)
    network = tflearn.conv_2d(network, 384, 3, activation='relu')
    network = tflearn.conv_2d(network, 384, 3, activation='relu')
    network = tflearn.conv_2d(network, 256, 3, activation='relu')
    network = tflearn.max_pool_2d(network, 3, strides=2)
    network = tflearn.layers.normalization.local_response_normalization(network)
    network = tflearn.layers.core.fully_connected(network, 4096, activation='tanh')
    network = tflearn.layers.core.dropout(network, 0.5)
    network = tflearn.layers.core.fully_connected(network, 4096, activation='tanh')
    network = tflearn.layers.core.dropout(network, 0.5)
    network = tflearn.layers.core.fully_connected(network, num_classes, activation='softmax')
    network = tflearn.layers.estimator.regression(network, optimizer='momentum',
                                                  loss='categorical_crossentropy',
                                                  learning_rate=0.001,
                                                  name='target')
    return network


def load_data():
    x, y = image_preloader('dataset_tropic/dataset.txt',
                           image_shape=(img_size, img_size),
                           mode='file',
                           categorical_labels=True,
                           normalize=True)
    return x, y


def training(network, x, y):
    model = tflearn.DNN(network, tensorboard_verbose=0)

    model.fit({'input': x}, {'target': y}, n_epoch=20,
              validation_set=0.1,
              snapshot_step=100, show_metric=True, run_id='convnet',
              batch_size=64, shuffle=True)


def main():
    """
    Main function to run convolutional neural network
    """
    x, y = load_data()
    network = alex_net()
    training(network=network, x=x, y=y)


if __name__ == '__main__':
    main()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorlayer as tl
from googlenet import google_net

import tflearn
from tflearn.data_utils import image_preloader
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

from spatial_transformer import transformer

tf.logging.set_verbosity(tf.logging.INFO)

# Global variables
img_size = 64
num_channels = 3
num_classes = 10


def cnn_net(img_aug):
    if img_aug:
        network = input_data(shape=[None, img_size, img_size, num_channels],
                             data_augmentation=img_aug,
                             name='input')
    else:
        network = input_data(shape=[None, img_size, img_size, num_channels],
                             name='input')

    network = conv_2d(network, 32, 3, activation='relu', name='conv1', regularizer='L2')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 64, 3, activation='relu', name='conv1', regularizer='L2')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, num_classes, activation='softmax')
    network = regression(network, optimizer='adam',
                         learning_rate=0.01,
                         loss='categorical_crossentropy',
                         name='target')
    return network


def alex_net(img_aug):
    if img_aug:
        network = input_data(shape=[None, img_size, img_size, num_channels],
                             data_augmentation=img_aug,
                             name='input')
    else:
        network = input_data(shape=[None, img_size, img_size, num_channels],
                             name='input')

    layers.transformer()
    network = conv_2d(network, 96, 11, strides=4, activation='relu', regularizer='L2')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, num_classes, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001,
                         name='target')
    return network


def res_net_1(img_aug):
    if img_aug:
        network = tflearn.input_data(shape=[None, img_size, img_size, num_channels],
                                     data_augmentation=img_aug,
                                     name='input')
    else:
        network = tflearn.input_data(shape=[None, img_size, img_size, num_channels],
                                     name='input')

    # Building Residual Network
    network = tflearn.conv_2d(network, 64, 3, activation='relu', bias=False)
    # Residual blocks
    network = tflearn.residual_bottleneck(network, 3, 16, 64)
    network = tflearn.residual_bottleneck(network, 1, 32, 128, downsample=True)
    network = tflearn.residual_bottleneck(network, 2, 32, 128)
    network = tflearn.residual_bottleneck(network, 1, 64, 256, downsample=True)
    network = tflearn.residual_bottleneck(network, 2, 64, 256)
    network = tflearn.batch_normalization(network)
    network = tflearn.activation(network, 'relu')
    network = tflearn.global_avg_pool(network)
    # Regression
    network = tflearn.fully_connected(network, num_classes, activation='softmax')
    network = tflearn.regression(network, optimizer='momentum',
                                 loss='categorical_crossentropy',
                                 learning_rate=0.001)
    return network


def res_net_2(img_aug):
    # Residual blocks
    # 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
    n = 5

    if img_aug:
        network = tflearn.input_data(shape=[None, img_size, img_size, num_channels],
                                     data_augmentation=img_aug,
                                     name='input')
    else:
        network = tflearn.input_data(shape=[None, img_size, img_size, num_channels],
                                     name='input')

    network = tflearn.conv_2d(network, 16, 3, regularizer='L2', weight_decay=0.0001)
    network = tflearn.residual_block(network, n, 16)
    network = tflearn.residual_block(network, 1, 32, downsample=True)
    network = tflearn.residual_block(network, n - 1, 32)
    network = tflearn.residual_block(network, 1, 64, downsample=True)
    network = tflearn.residual_block(network, n - 1, 64)
    network = tflearn.batch_normalization(network)
    network = tflearn.activation(network, 'relu')
    network = tflearn.global_avg_pool(network)
    # Regression
    network = tflearn.fully_connected(network, num_classes, activation='softmax')
    mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
    network = tflearn.regression(network, optimizer=mom,
                                 loss='categorical_crossentropy',
                                 name='target')
    return network


def load_data(train=True):
    if train:
        target_path = 'dataset_tropic/dataset.txt'
    else:
        target_path = 'dataset_tropic/dataseteval.txt'
    x, y = image_preloader(target_path=target_path,
                           image_shape=(img_size, img_size),
                           mode='file',
                           categorical_labels=True,
                           normalize=True)
    return x, y


def data_augmentation(index):
    img_aug = tflearn.ImageAugmentation()
    if index is 1:
        img_aug.add_random_flip_leftright()
    elif index is 2:
        img_aug.add_random_blur(sigma_max=2.5)
    elif index is 3:
        img_aug.add_random_90degrees_rotation(rotations=[0, 1, 2, 3])
    elif index is 4:
        img_aug.add_random_rotation(max_angle=10)
    return img_aug


def training(network):
    model = tflearn.DNN(network, tensorboard_verbose=0)
    x, y = load_data()

    # model.load("model.tfl")
    model.fit(x, y, n_epoch=150,
              validation_set=0.1,
              snapshot_step=100, snapshot_epoch=False,
              show_metric=True, run_id='convnet',
              batch_size=64, shuffle=True)
    model.save("./checkpoints/model.tfl")


def evaluate(network):
    model = tflearn.DNN(network, tensorboard_verbose=0)
    x, y = load_data(train=False)

    model.load("./checkpoints/model.tfl")
    score = model.evaluate(x, y, batch_size=128)
    print('Test accuracy: %0.4f%%' % (score[0] * 100))

    prediction = model.predict([x[0]])
    print("Prediction: %s" % str(prediction[0]))


def main():
    """
    Main function to run convolutional neural network
    """
    img_aug = data_augmentation(index=2)

    network = cnn_net(img_aug=img_aug)
    # network = alex_net(img_aug=img_aug)
    # network = google_net(img_aug=img_aug)
    # network = res_net_1(img_aug=img_aug)
    # network = res_net_2(img_aug=img_aug)

    training(network=network)

    # evaluate(network=network)


if __name__ == '__main__':
    main()

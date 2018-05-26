from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras

print(keras.__version__)
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Input, Convolution2D
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from spatial_transformer import SpatialTransformer

import numpy as np
from tflearn.data_utils import image_preloader

from mixup_generator import MixupGenerator

# Global variables
img_size = 224
num_channels = 3
num_classes = 10
batch_size = 64
num_epochs = 20


def load_data(train=True):
    if train:
        target_path = 'dataset_tropic/dataset.txt'
    else:
        target_path = 'dataset_tropic/dataseteval.txt'
    x, y = image_preloader(target_path=target_path,
                           image_shape=(img_size, img_size),
                           mode='file',
                           categorical_labels=True,
                           normalize=True,
                           grayscale=False)
    x = np.asarray(x[:])
    # Resize to change image values from range 0,1 to 0,255
    # x *= 255
    y = np.asarray(y[:])
    return x, y


def spatial_transformer(images):
    input_shape = images.shape[1:]
    print(input_shape)
    # initial weights
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((50, 6), dtype='float32')
    weights = [W, b.flatten()]

    locnet = Sequential()
    locnet.add(MaxPooling2D(pool_size=(2, 2), input_shape=input_shape))
    locnet.add(Convolution2D(20, 5, 5))
    locnet.add(MaxPooling2D(pool_size=(2, 2)))
    locnet.add(Convolution2D(20, 5, 5))

    locnet.add(Flatten())
    locnet.add(Dense(50))
    # locnet.add(Dropout(.5))
    locnet.add(Activation('relu'))
    locnet.add(Dense(6, weights=weights))
    # locnet.add(Activation('sigmoid'))

    model = Sequential()
    model.add(SpatialTransformer(localization_net=locnet,
                                 downsample_factor=1,
                                 input_shape=input_shape))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def simple_cnn(images):
    images = images.shape[1:]
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=images))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def alex_net():
    model = Sequential()
    # model.add(Conv2D(96, (11,11), strides=(4,4), activation='relu', padding='same', input_shape=(img_height, img_width, channel,)))
    # for original Alexnet
    model.add(Conv2D(96, (3, 3), strides=(2, 2), activation='relu', padding='same',
                     input_shape=(img_size, img_size, num_channels,)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # Local Response normalization for Original Alexnet
    model.add(BatchNormalization())

    model.add(Conv2D(256, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # Local Response normalization for Original Alexnet
    model.add(BatchNormalization())

    model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # Local Response normalization for Original Alexnet
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def cnn_model(images, conv_net=None):
    input_tensor = Input(shape=(img_size, img_size, num_channels))

    # cnn_model == ResNet50
    if conv_net == 1:
        model = ResNet50(input_tensor=input_tensor,
                         weights=None,
                         include_top=True,
                         classes=num_classes)
    # cnn_model == InceptionV3 (GoogleNet)
    elif conv_net == 2:
        model = InceptionV3(input_tensor=input_tensor,
                            weights=None,
                            include_top=True,
                            classes=num_classes)
    # cnn_model == AlexNet
    elif conv_net == 3:
        model = alex_net()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model


def training(use_data_aug=True, use_mixup=False):
    x_train, y_train = load_data(train=True)
    x_test, y_test = load_data(train=False)

    # model = simple_cnn(x_train)

    # ConvNet models:
    # (cnn_model = 1) == ResNet50
    # (cnn_model = 2) == Inception v3 (GoogleNet)
    # (cnn_model = 3) == AlexNet

    # model = cnn_model(x_train, conv_net=1)
    # model = cnn_model(x_train, conv_net=2)
    model = cnn_model(x_train, conv_net=3)
    # model = spatial_transformer(x_train)

    if use_mixup:
        datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)
        training_generator = MixupGenerator(x_train, y_train,
                                            batch_size=batch_size,
                                            alpha=0.2,
                                            datagen=datagen)()
        model.fit_generator(generator=training_generator,
                            epochs=num_epochs,
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            validation_data=(x_test, y_test),
                            shuffle=True)
    if not use_data_aug and not use_mixup:
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=num_epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    elif use_data_aug:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        datagen.fit(x_train)
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            epochs=num_epochs,
                            validation_data=(x_test, y_test),
                            workers=4)
    model.save('my_model.h5')
    del model


def evaluate(model):
    model = load_model('my_model.h5')
    x_test, y_test = load_data(train=False)
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


def main():
    """
    Main function to run convolutional neural network
    """
    training(use_data_aug=False, use_mixup=False)


if __name__ == '__main__':
    main()

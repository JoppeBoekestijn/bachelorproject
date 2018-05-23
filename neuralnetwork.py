from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.applications.resnet50 import ResNet50, preprocess_input


import numpy as np
from tflearn.data_utils import image_preloader

tf.logging.set_verbosity(tf.logging.INFO)

# Global variables
img_size = 128
num_channels = 3
num_classes = 10
batch_size = 32
num_epochs = 20
input_shape = (img_size, img_size, num_channels)


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
    x *= 255
    y = np.asarray(y[:])
    return x, y


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


def resnet50(images):
    input_tensor = Input(shape=(img_size, img_size, num_channels))
    model = ResNet50(input_tensor=input_tensor,
                     weights=None,
                     include_top=True,
                     classes=num_classes)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model


def training(data_aug=True):
    x_train, y_train = load_data(train=True)
    x_test, y_test = load_data(train=False)

    # model = simple_cnn(x_train)
    model = resnet50(x_train)

    if not data_aug:
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=num_epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    elif data_aug:
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
    training(data_aug=False)


if __name__ == '__main__':
    main()





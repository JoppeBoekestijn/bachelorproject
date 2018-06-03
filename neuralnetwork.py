from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras
from keras import backend as k

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Input, Convolution2D
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3

import numpy as np
from tflearn.data_utils import image_preloader

from mixup_generator import MixupGenerator
from cutout import get_random_cutout
from res_net import resnet_v1, resnet_v2
# from resnetcifarexample import resnet_v1, resnet_v2

# Global variables
img_size = 224
num_channels = 3
num_classes = 10
batch_size = 20
num_epochs = 200

# Dataset
train_dir = './dataset_tropic/train'
test_dir = './dataset_tropic/test'


def load_data(train=True, subtract_pixel_mean=True):
    # if train:
    #     target_path = 'dataset_tropic/dataset.txt'
    # elif not train:
    #     target_path = 'dataset_tropic/dataseteval.txt'
    if train:
        target_path = train_dir
    elif not train:
        target_path = test_dir
    x, y = image_preloader(target_path=target_path,
                           image_shape=(img_size, img_size),
                           mode='folder',
                           categorical_labels=True,
                           normalize=True,
                           grayscale=False)
    x = np.asarray(x[:])
    # Resize to change image values from range 0,1 to 0,255
    # x *= 255
    y = np.asarray(y[:])

    if subtract_pixel_mean:
        x_mean = np.mean(x, axis=0)
        x -= x_mean

    # Shuffle the data so later on not only the last classes are used
    # in the validation set
    x, y = random_shuffle(x, y)
    return x, y


def random_shuffle(images, labels):
    perm = np.arange(len(images))
    np.random.shuffle(perm)
    random_image = images[perm]
    random_label = labels[perm]
    return random_image, random_label


def alex_net():
    model = Sequential()

    model.add(Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11, 11),
                     strides=(4, 4), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(4096, input_shape=(224 * 224 * 3,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model


def cnn_model(images, conv_net=None):
    input_tensor = Input(shape=(img_size, img_size, num_channels))
    # Different input shape needed for ResNet_v1 and ResNet_v2
    # Since they call input function themselves
    input_shape = (img_size, img_size, num_channels)
    n = 3

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

    # cnn_model == ResNet_v1
    elif conv_net == 4:
        model = resnet_v1(input_shape=input_shape,
                          depth=n * 6 + 2,
                          num_classes=10)

    # cnn_model == ResNet_v2
    elif conv_net == 5:
        model = resnet_v2(input_shape=input_shape,
                          depth=n * 9 + 2,
                          num_classes=10)

    model.summary()
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model


# def lr_schedule(epoch):
#     lr = 1e-3
#     if epoch > 180:
#         lr *= 0.5e-3
#     elif epoch > 160:
#         lr *= 1e-3
#     elif epoch > 120:
#         lr *= 1e-2
#     elif epoch > 80:
#         lr *= 1e-1
#     print('Learning rate: ', lr)
#     return lr


def training(use_data_aug=False, use_mixup=False, use_cutout=False):
    x_train, y_train = load_data(train=True)
    x_test, y_test = load_data(train=False)

    # model = simple_cnn(x_train)

    # ConvNet models:
    # (cnn_model = 1) == ResNet50
    # (cnn_model = 2) == Inception v3 (GoogleNet)
    # (cnn_model = 3) == AlexNet
    # (cnn_model = 4) == ResNet_v1
    # (cnn_model = 5) == ResNet_v2

    model = cnn_model(x_train, conv_net=4)

    checkpoint = ModelCheckpoint(filepath='./models/test.h5',
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=3,
                                   min_lr=0.5e-6)
    callbacks = [checkpoint]

    if use_mixup:
        datagen = ImageDataGenerator(
            width_shift_range=0.0,
            height_shift_range=0.0,
            horizontal_flip=False)
        training_generator = MixupGenerator(x_train, y_train,
                                            batch_size=batch_size,
                                            alpha=0.2,
                                            datagen=datagen)()
        model.fit_generator(generator=training_generator,
                            epochs=num_epochs,
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            validation_data=(x_test, y_test),
                            shuffle=True,
                            callbacks=callbacks)
    if use_cutout:
        datagen = ImageDataGenerator(
            width_shift_range=0,
            height_shift_range=0,
            horizontal_flip=False,
            preprocessing_function=get_random_cutout(v_l=0, v_h=255))
        datagen.fit(x_train)
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            epochs=num_epochs,
                            validation_data=(x_test, y_test),
                            workers=4,
                            callbacks=callbacks)
    if not use_data_aug and not use_mixup:
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=num_epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks)
    elif use_data_aug:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        datagen.fit(x_train)
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            epochs=num_epochs,
                            validation_data=(x_test, y_test),
                            workers=4,
                            callbacks=callbacks)
    # model.save('./models/model.h5')
    # del model


def evaluate():
    model = load_model('./models/googlenet_true_false_100.h5')
    x_test, y_test = load_data(train=False)
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


def main():
    """
    Main function to run convolutional neural network
    """
    config = tf.ConfigProto()

    config.gpu_options.allow_growth = True

    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    k.tensorflow_backend.set_session(tf.Session(config=config))

    training(use_data_aug=False, use_mixup=False, use_cutout=False)

    # evaluate()


if __name__ == '__main__':
    main()

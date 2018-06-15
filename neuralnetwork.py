from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

import tensorflow as tf
import keras
from keras import backend as k

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.callbacks import TensorBoard
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16

import numpy as np
from tflearn.data_utils import image_preloader

from mixup_generator import MixupGenerator
from cutout import get_random_cutout
from res_net import resnet_v1, resnet_v2

# Global variables
img_size = 224 # 224 standard
num_channels = 3
num_classes = 10
batch_size = 20 # 20 standard
num_epochs = 50

# Dataset
train_dir = './dataset_tropic/train'
test_dir = './dataset_tropic/test'


def load_data(train=True, subtract_pixel_mean=True):
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


def cnn_model(images, conv_net=None):
    input_tensor = Input(shape=(img_size, img_size, num_channels))
    # Different input shape needed for ResNet_v1 and ResNet_v2
    # Since they call input function themselves
    input_shape = (img_size, img_size, num_channels)

    # cnn_model == ResNet50
    if conv_net == 1:
        # model = ResNet50(input_tensor=input_tensor,
        #                  weights='imagenet',
        #                  include_top=False,
        #                  input_shape=(img_size, img_size, num_channels),
        #                  classes=num_classes)
        base_model = ResNet50(weights='imagenet',
                              include_top=False,
                              input_shape=(img_size, img_size, num_channels))
        model = base_model.output
        # model = GlobalAveragePooling2D()(model)
        model = Flatten()(model)
        model = Dense(num_classes, activation='softmax')(model)
        model = Model(input=base_model.input,
                      output=model)
    elif conv_net == 2:
        base_model = InceptionV3(weights='imagenet',
                                 include_top=False,
                                 input_shape=(img_size, img_size, num_channels))
        model = base_model.output
        model = GlobalAveragePooling2D()(model)
        x = Dense(1024, activation='relu')(model)
        # model = Flatten()(model)
        model = Dense(num_classes, activation='softmax')(model)
        model = Model(input=base_model.input,
                      output=model)

    # cnn_model == VGG16
    elif conv_net == 3:
        base_model = VGG16(weights='imagenet',
                           include_top=False,
                           input_shape=(img_size, img_size, num_channels))
        model = base_model.output
        # model = GlobalAveragePooling2D()(model)
        model = Flatten()(model)
        model = Dense(num_classes, activation='softmax')(model)
        model = Model(input=base_model.input,
                      output=model)

    # cnn_model == InceptionV3(GoogleNet) from scratch
    elif conv_net == 4:
        model = InceptionV3(input_tensor=input_tensor,
                            weights=None,
                            include_top=True,
                            classes=num_classes)
    # cnn_model == ResNet from scratch
    elif conv_net == 5:
        model = ResNet50(input_tensor=input_tensor,
                         weights=None,
                         include_top=True,
                         classes=num_classes)

    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=lr_schedule(0)),
                  metrics=['accuracy'])
    return model


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 80:
        lr *= 0.5e-3
    elif epoch > 60:
        lr *= 1e-3
    elif epoch > 40:
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def training(filepath, use_data_aug=False, use_mixup=False, use_cutout=False):
    # Load the training and test data
    x_train, y_train = load_data(train=True)
    x_test, y_test = load_data(train=False)

    # Add a Tensorboard callback for visualization using Tensorboard
    tensorboard = TensorBoard(log_dir='./logs/' + filepath,
                              histogram_freq=0,
                              write_graph=True,
                              write_images=False)

    # ConvNet models:
    # (cnn_model = 1) == ResNet50
    # (cnn_model = 2) == Inception v3 (GoogleNet)
    # (cnn_model = 3) == AlexNet
    # (cnn_model = 4) == ResNet_v1
    # (cnn_model = 5) == ResNet_v2
    # (cnn_model = 6) == VGG-16

    model = cnn_model(x_train, conv_net=4)

    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    # lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
    #                                cooldown=0,
    #                                patience=5,
    #                                min_lr=0.5e-6)

    callbacks = [checkpoint, lr_scheduler, tensorboard]

    if use_mixup:
        datagen = ImageDataGenerator()
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
    elif use_cutout:
        datagen = ImageDataGenerator(preprocessing_function=get_random_cutout(v_l=0, v_h=1, pixel_level=False))
        datagen.fit(x_train)
        model.fit_generator(generator=datagen.flow(x_train, y_train,
                                                   batch_size=batch_size),
                            epochs=num_epochs,
                            validation_data=(x_test, y_test),
                            shuffle=True,
                            callbacks=callbacks)
    elif use_data_aug:
        datagen = ImageDataGenerator(
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        datagen.fit(x_train)
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            epochs=num_epochs,
                            validation_data=(x_test, y_test),
                            shuffle=True,
                            callbacks=callbacks)
    # No data augmentation
    else:
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=num_epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks)
    # model.save('./models/model.h5')
    # del model


def evaluate(filepath):
    # Load the test data and evaluate once without data augmentation
    model = load_model(filepath=filepath)
    x_test, y_test = load_data(train=False)
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


def main(filepath):
    """
    Main function to run convolutional neural network
    """
    # Config to not overload GPU
    config = tf.ConfigProto()

    config.gpu_options.allow_growth = True

    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    k.tensorflow_backend.set_session(tf.Session(config=config))

    # Instantiate the training with chosen setting
    # training(filepath='./models/googlenet_widthshift20_50.h5',
    #          use_data_aug=True,
    #          use_mixup=False,
    #          use_cutout=False)

    training(filepath=filepath,
             use_data_aug=True,
             use_mixup=False,
             use_cutout=False)
    # Evaluate model on test data once, without any augmentation
    # evaluate(filepath='./models/googlenet_widthshift20_50_scratch.h5')


if __name__ == '__main__':
    filepath = sys.argv[1]
    main(filepath=filepath)


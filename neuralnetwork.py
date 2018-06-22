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

# Global variables
img_size = 224  # 224 standard
num_channels = 3
num_classes = 10
batch_size = 20  # 20 standard
num_epochs = 50

# Dataset
train_dir = './dataset_tropic/exp3/train'
test_dir = './dataset_tropic/exp3/test'


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

    # cnn_model == ResNet50 with pre-trained weights
    if conv_net == 1:
        base_model = ResNet50(weights='imagenet',
                              include_top=False,
                              input_shape=(img_size, img_size, num_channels))
        model = base_model.output
        # model = GlobalAveragePooling2D()(model)
        model = Flatten()(model)
        model = Dense(num_classes, activation='softmax')(model)
        model = Model(input=base_model.input,
                      output=model)
    # cnn-model == InceptionV3(GoogleNet) with pre-trained weights
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
    # cnn_model == InceptionV3(GoogleNet) from scratch
    elif conv_net == 3:
        model = InceptionV3(input_tensor=input_tensor,
                            weights=None,
                            include_top=True,
                            classes=num_classes)
    # cnn_model == ResNet from scratch
    elif conv_net == 4:
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


def use_data_aug(callbacks, model, x_train, y_train, x_test, y_test, data_aug=None):
    # Rotation
    if data_aug == 1:
        datagen = ImageDataGenerator(
            rotation_range=60)  # randomly rotate images in the range (degrees, 0 to 180)
    # Horizontal flip
    elif data_aug == 2:
        datagen = ImageDataGenerator(
            horizontal_flip=True)  # randomly flip images
    # Vertical flip
    elif data_aug == 3:
        datagen = ImageDataGenerator(
            vertical_flip=True)  # randomly flip images
    # Horizontal shift + horizontal flip
    elif data_aug == 4:
        datagen = ImageDataGenerator(
            height_shift_range=0.2,
            horizontal_flip=True)  # randomly shift images vertically (fraction of total height)
    # Vertical shift + vertical flip
    elif data_aug == 5:
        datagen = ImageDataGenerator(
            width_shift_range=0.2,
            vertical_flip=True)  # randomly shift images horizontally (fraction of total width)

    datagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=num_epochs,
                        validation_data=(x_test, y_test),
                        shuffle=True,
                        callbacks=callbacks)


def advanced_data_aug(callbacks, model, x_train, y_train, x_test, y_test, data_aug=None):
    # Apply mix-up
    if data_aug == 6:
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
    # Apply cut_out + vertical flip + horizontal shift
    elif data_aug == 7:
        datagen = ImageDataGenerator(preprocessing_function=get_random_cutout(v_l=0, v_h=1, pixel_level=False),
                                     vertical_flip=True,
                                     height_shift_range=0.2)
        datagen.fit(x_train)
        model.fit_generator(generator=datagen.flow(x_train, y_train,
                                                   batch_size=batch_size),
                            epochs=num_epochs,
                            validation_data=(x_test, y_test),
                            shuffle=True,
                            callbacks=callbacks)
    # No data augmentation
    elif data_aug == 8:
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=num_epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks)


def training(filepath, conv_net=None, data_aug=None):
    # Load the training and test data
    x_train, y_train = load_data(train=True)
    x_test, y_test = load_data(train=False)

    # Add a Tensorboard callback for visualization using Tensorboard
    tensorboard = TensorBoard(log_dir='./logs/' + filepath,
                              histogram_freq=0,
                              write_graph=True,
                              write_images=False)

    conv_net = int(conv_net)
    data_aug = int(data_aug)

    if conv_net == 1:
        model = cnn_model(x_train, conv_net=1)
    elif conv_net == 2:
        model = cnn_model(x_train, conv_net=2)
    elif conv_net == 3:
        model = cnn_model(x_train, conv_net=3)
    else:
        model = cnn_model(x_train, conv_net=4)

    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    callbacks = [checkpoint, lr_scheduler, tensorboard]

    # Apply traditional data augmentation
    if data_aug < 6:
        use_data_aug(callbacks=callbacks,
                     model=model,
                     data_aug=data_aug,
                     x_train=x_train,
                     y_train=y_train,
                     x_test=x_test,
                     y_test=y_test)
    elif data_aug > 5:
        advanced_data_aug(callbacks=callbacks,
                          model=model,
                          data_aug=data_aug,
                          x_train=x_train,
                          y_train=y_train,
                          x_test=x_test,
                          y_test=y_test)


def evaluate(filepath):
    # Load the test data and evaluate once without data augmentation
    model = load_model(filepath=filepath)
    x_test, y_test = load_data(train=False)
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


def main(filepath, conv_net, data_aug):
    """
    Main function to run convolutional neural network
    """
    # Config to not overload GPU
    config = tf.ConfigProto()

    config.gpu_options.allow_growth = True

    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    k.tensorflow_backend.set_session(tf.Session(config=config))

    print(filepath)
    print(conv_net)
    print(data_aug)

    # Instantiate the training with chosen setting
    # training(filepath='./models/comb/googlenet_vertflip_vertshift_scratch.h5',
    #          use_data_aug=True,
    #          use_mixup=False,
    #          use_cutout=False)

    filepath = "./models/exp3/comb/" + filepath

    # ConvNet models:
    # (cnn_model = 1) == ResNet50 with pre-trained weights
    # (cnn_model = 2) == Inception v3 (GoogleNet) with pre-trained weights
    # (cnn_model = 3) == Inception v3 (GoogleNet) from scratch
    # (cnn_model = 4) == ResNet50 from scratch

    # Data augmentation:
    # (data_aug = 1) == Rotation
    # (data_aug = 2) == Horizontal flip
    # (data_aug = 3) == Vertical flip
    # (data_aug = 4) == Horizontal shift
    # (data_aug = 5) == Vertical shift
    # (data_aug = 6) == Mix-up
    # (data_aug = 7) == Cutout
    # (data_agu = 8) == No data augmentation

    training(filepath=filepath,
             conv_net=conv_net,
             data_aug=data_aug)

    # Evaluate model on test data once, without any augmentation
    # evaluate(filepath='./models/exp3/single/3googlenet_cutout_pre.h5')


if __name__ == '__main__':
    filepath = sys.argv[1]
    conv_net = sys.argv[2]
    data_aug = sys.argv[3]
    main(filepath=filepath, conv_net=conv_net, data_aug=data_aug)

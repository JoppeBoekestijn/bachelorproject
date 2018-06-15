#
#
# def spatial_transformer(images):
#     input_shape = images.shape[1:]
#     print(input_shape)
#     # initial weights
#     b = np.zeros((2, 3), dtype='float32')
#     b[0, 0] = 1
#     b[1, 1] = 1
#     W = np.zeros((50, 6), dtype='float32')
#     weights = [W, b.flatten()]
#
#     locnet = Sequential()
#     locnet.add(MaxPooling2D(pool_size=(2, 2), input_shape=input_shape))
#     locnet.add(Convolution2D(20, 5, 5))
#     locnet.add(MaxPooling2D(pool_size=(2, 2)))
#     locnet.add(Convolution2D(20, 5, 5))
#
#     locnet.add(Flatten())
#     locnet.add(Dense(50))
#     # locnet.add(Dropout(.5))
#     locnet.add(Activation('relu'))
#     locnet.add(Dense(6, weights=weights))
#     # locnet.add(Activation('sigmoid'))
#
#     model = Sequential()
#     model.add(SpatialTransformer(localization_net=locnet,
#                                  downsample_factor=1,
#                                  input_shape=input_shape))
#
#     model.add(Convolution2D(32, 3, 3, border_mode='same'))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Convolution2D(32, 3, 3))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Flatten())
#     model.add(Dense(256))
#     model.add(Activation('relu'))
#
#     model.add(Dense(num_classes))
#     model.add(Activation('softmax'))
#
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
#     return model
#
#
# def simple_cnn(images):
#     images = images.shape[1:]
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), padding='same',
#                      input_shape=images))
#     model.add(Activation('relu'))
#     model.add(Conv2D(32, (3, 3)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Conv2D(64, (3, 3), padding='same'))
#     model.add(Activation('relu'))
#     model.add(Conv2D(64, (3, 3)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Flatten())
#     model.add(Dense(512))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(num_classes))
#     model.add(Activation('softmax'))
#
#     model.compile(loss=keras.losses.categorical_crossentropy,
#                   optimizer=keras.optimizers.Adadelta(),
#                   metrics=['accuracy'])
#     return model

#
#
# def alex_net():
#     model = Sequential()
#
#     model.add(Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11, 11),
#                      strides=(4, 4), padding='valid'))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
#     model.add(BatchNormalization())
#
#     model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
#     model.add(BatchNormalization())
#
#     model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
#     model.add(Activation('relu'))
#     model.add(BatchNormalization())
#
#     model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
#     model.add(Activation('relu'))
#     model.add(BatchNormalization())
#
#     model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
#     model.add(BatchNormalization())
#
#     model.add(Flatten())
#     model.add(Dense(4096, input_shape=(224 * 224 * 3,)))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     model.add(BatchNormalization())
#
#     model.add(Dense(4096))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     model.add(BatchNormalization())
#
#     model.add(Dense(1000))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     model.add(BatchNormalization())
#
#     model.add(Dense(10))
#     model.add(Activation('softmax'))
#
#     return model
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# matplotlib inline
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import glob
tf.logging.set_verbosity(tf.logging.INFO)

# Datapath
data_path = 'dataset_tropic'
train_files = glob.glob(data_path + '/' + 'train-*')
evaluate_files = glob.glob(data_path + '/' + 'validation-*')

# Global variables
# Convolutional layer 1
filter_size1 = 5
num_filters1 = 16

# Convolutional layer 2
filter_size2 = 5
num_filters2 = 36

# Fully-connected layer
fc_size = 128

# Data Dimensions
img_size = 64
num_channels = 3
num_classes = 10


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))





def new_weights(shape):
    """

    :param shape:
    :return: define variable to represent weights
    """
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    """

    :param length:
    :return: define variable to represent biases
    """
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(input,  # Previous layer
                   num_input_channels,  # Number of channels in previous layer
                   filter_size,  # Width and height of each filter
                   num_filters,  # Number of filters
                   use_pooling=True):  # Use 2x2 max-pooling
    """

    :param input:
    :param num_input_channels:
    :param filter_size:
    :param num_filters:
    :param use_pooling:
    :return:
    """
    # Shape of the filter-weights for the convolution
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights with the given shape
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter
    biases = new_biases(length=num_filters)

    # Tensorflow operation for convolution
    # First and last stride must always be 1
    # Padding is set to 'SAME' which means the input image
    # is padded with zeros so the size of the output is the same
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add biases to the results of the convolution
    # A bias-value is added to each filter-channel
    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Apply a ReLu activation function
    layer = tf.nn.relu(layer)

    return layer, weights


def new_fc_layer(input,
                 num_inputs,
                 num_outputs,
                 use_relu=True):
    """

    :param input:
    :param num_inputs:
    :param num_outputs:
    :param use_relu:
    :return:
    """
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of the input
    # and weights, and the add the bias-values
    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


def flatten_layer(layer):
    """

    :param layer:
    :return:
    """
    # Get the shape of the input layer
    layer_shape = layer.get_shape()

    # Number of features is: img_size * img_width * num_channels
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features


def read_and_decode(serialized):
    """

    :param serialized:
    :return:
    """
    features = {'image/encoded': tf.FixedLenFeature([], tf.string),
                'image/class/text': tf.FixedLenFeature([], tf.string),
                'image/class/label': tf.FixedLenFeature([], tf.int64)}

    # print('image/encoded: {}'.format(features['image/encoded']))

    features = tf.parse_single_example(serialized=serialized, features=features)
    # print('image/encoded/parsed: {}'.format(features['image/encoded']))
    image = tf.decode_raw(features['image/encoded'], tf.uint8)

    image = tf.reshape(image, [250, 250, 3])

    resized_image = tf.image.resize_image_with_crop_or_pad(image, 64, 64)

    resized_image = tf.cast(resized_image, tf.float32)

    # Trying something new
    # image = tf.reshape(features['image/encoded'], shape=[])

    # image = tf.decode_raw(image_raw, tf.float32)
    # image = tf.image.decode_image(image, channels=3)
    # print('image decoded: {}'.format(image))
    #
    # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # print('image: {}'.format(image))
    #
    # # image = tf.cast(image, tf.float32)
    # #
    # resized_image = tf.reshape(image, [img_size, img_size, num_channels])

    # image = tf.reshape(image, [img_size * img_size * num_channels])
    # print('image reshape: {}'.format(image))

    label = features['image/class/label']

    # The labels numbered from 1 to 10 representing the different classes
    # label = tf.cast(tf.reshape(features['image/class/label'], shape=[]), dtype=tf.int64)
    #
    # image = tf.image.decode_and_crop_jpeg(
    #     contents=features['image/encoded'],
    #     channels=num_channels,
    #     crop_window=[0, 0, img_size, img_size])
    #
    # image = (tf.image.convert_image_dtype(image, dtype=tf.float32))

    return {'image': resized_image}, label


# def read_and_decode_preprocess(serialized):
#     """
#
#     :param serialized:
#     :return:
#     """
#     features = {'image/encoded': tf.FixedLenFeature([], tf.string),
#                 'image/class/text': tf.FixedLenFeature([], tf.string),
#                 'image/class/label': tf.FixedLenFeature([], tf.int64)}
#
#     features = tf.parse_single_example(serialized, features=features)
#
#     # The labels numbered from 1 to 10 representing the different classes
#     label = features['image/class/label']
#
#     image = tf.image.decode_and_crop_jpeg(
#         contents=features['image/encoded'],
#         channels=num_channels,
#         crop_window=[0, 0, img_size, img_size])
#
#     flipped_image = tf.image.random_flip_left_right(image)
#
#     flipped_image = (tf.image.convert_image_dtype(image, dtype=tf.float32))
#
#     return flipped_image, label


# def inputs(batch_size, num_epochs):
#     """
#
#     :param batch_size:
#     :param num_epochs:
#     :return:
#     """
#     if not num_epochs:
#         num_epochs = None
#
#     filename_queue = tf.train.string_input_producer(train_files, num_epochs)
#     reader = tf.TFRecordReader()
#     _, serialized = reader.read(filename_queue)
#
#     image, label = read_and_decode(serialized)
#
#     image_batch, label_batch = tf.train.shuffle_batch(
#         [image, label],
#         batch_size=batch_size,
#         capacity=1000 + 3 * batch_size,
#         num_threads=2,
#         min_after_dequeue=1000)
#
#     return image_batch, label_batch
#
#
# def printing_tfrecords():
#     """
#     Iterate through filenames and print result
#     """
#     for example in tf.python_io.tf_record_iterator(train_files):
#         result = tf.train.Example.FromString(example)
#         print(result)
#
#
# def plot_images():
#     """
#     Plots and displays an image
#     """
#     image, label = inputs(10, num_epochs=None)
#
#     sess = tf.Session()
#
#     with sess.as_default():
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(coord=coord)
#
#         for i in range(1):
#             img, labeltext = sess.run([image, label])
#             img = img.astype(np.uint8)
#             for j in range(1):
#                 plt.imshow(img[j])
#                 plt.title(labeltext[j])
#                 plt.show()
#
#         # Stop the threads
#         coord.request_stop()
#
#         # Wait for threads to stop
#         coord.join(threads)


def input_fn(train, batch_size=32, buffer_size=10000):
    """

    :param train:
    :param batch_size:
    :param buffer_size:
    :return:
    """
    if train:
        dataset = tf.data.TFRecordDataset(filenames=train_files)
        # dataset_flipped = tf.data.TFRecordDataset(filenames=train_files)
    else:
        dataset = tf.data.TFRecordDataset(filenames=evaluate_files)
        # dataset_flipped = tf.data.TFRecordDataset(filenames=evaluate_files)

    dataset = dataset.map(read_and_decode)
    # dataset_flipped = dataset_flipped.map(read_and_decode_preprocess)
    # dataset.concatenate(dataset_flipped)
    print('dataset: {}'.format(dataset))

    if train:
        dataset = dataset.shuffle(buffer_size=buffer_size)
        num_repeat = None
    else:
        num_repeat = 1

    # dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)

    # Might be neccessary to remove remainder in last batch
    # dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.repeat(num_repeat)
    iterator = dataset.make_one_shot_iterator()

    features, labels = iterator.get_next()
    print("features: {}".format(features))
    # print("labels_batch: {}".format(labels_batch))

    # images_batch = {'image': images_batch}

    return features, labels


def train_input_model():
    """

    :return:
    """
    return input_fn(train=True)


def test_input_model():
    """

    :return:
    """
    return input_fn(train=False)


def cnn_model_test(features, labels, mode):
    # x = features['image']
    # print('x shape: {}'.format(x.shape))
    # input_layer = tf.reshape(x, [32, 64, 64, 3])
    # input_layer = tf.reshape(input_layer, [-1, 64, 64, 3])
    # print("features: {}".format(features))
    # print("labels: {}".format(labels))
    x = features['image']
    input_layer = tf.reshape(x, [-1, img_size, img_size, num_channels])
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.leaky_relu)
    print("layer_conv1: {}".format(conv1))

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    print("pool1: {}".format(pool1))

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.leaky_relu)
    print("layer_conv2: {}".format(conv2))
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    print("pool2: {}".format(pool2))

    # Dense Layer
    # pool2_flat = tf.reshape(pool2, [-1, 16 * 16 * 64])
    pool2_flat, num = flatten_layer(pool2)

    print("pool2_flat: {}".format(pool2_flat))
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.leaky_relu)
    print("dense: {}".format(dense))
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    print("dropout: {}".format(dropout))

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)
    print("logits: {}".format(logits))

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    print("onehot_labels: {}".format(onehot_labels))
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# def cnn_model(features, labels, mode):
#     # Initialize first convolutional layer
#     layer_conv1, weights_conv1 = new_conv_layer(input=features,
#                                                 num_input_channels=num_channels,
#                                                 filter_size=filter_size1,
#                                                 num_filters=num_filters1,
#                                                 use_pooling=True)
#     print("layer_conv1: {}".format(layer_conv1))
#
#     # Initialize second convolutional layer
#     layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
#                                                 num_input_channels=num_filters1,
#                                                 filter_size=filter_size2,
#                                                 num_filters=num_filters2,
#                                                 use_pooling=True)
#     print("layer_conv2: {}".format(layer_conv2))
#
#     layer_flat, num_features = flatten_layer(layer_conv2)
#     print("layer_flat: {}".format(layer_flat.shape))
#
#     # Fully-connected layer
#     layer_fc1 = new_fc_layer(input=layer_flat,
#                              num_inputs=num_features,
#                              num_outputs=fc_size,
#                              use_relu=True)
#     print("layer_fc1: {}".format(layer_fc1.shape))
#
#     # Second fully-connected layer
#     layer_fc2 = new_fc_layer(input=layer_fc1,
#                              num_inputs=fc_size,
#                              num_outputs=num_classes,
#                              use_relu=False)
#     print("layer_fc2: {}".format(layer_fc2.shape))
#
#     # Normalize prediction using softmax
#     y_pred = tf.nn.softmax(layer_fc2)
#
#     # Predicted class is index with largest number
#     y_pred_cls = tf.argmax(y_pred, axis=1)
#
#     # onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
#
#     # Calculate the overall loss
#     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=layer_fc2,
#                                                                    labels=labels)
#     loss = tf.reduce_mean(cross_entropy)
#
#     predictions = {
#         "classes": y_pred_cls,
#         "probabilities": tf.nn.softmax(layer_fc2, name="softmax_tensor")
#     }
#
#     # if mode == tf.estimator.ModeKeys.PREDICT:
#         # return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
#
#     if mode == tf.estimator.ModeKeys.TRAIN:
#         optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
#         train_op = optimizer.minimize(
#             loss=loss,
#             global_step=tf.train.get_global_step())
#         return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
#
#     eval_metric_ops = {
#         "accuracy": tf.metrics.accuracy(
#             labels=labels, predictions=predictions["classes"])}
#     return tf.estimator.EstimatorSpec(
#         mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    """
    Main function to run convolutional neural network
    """
    # Function to plot one image
    # plot_images()

    classifier = tf.estimator.Estimator(model_fn=cnn_model_test, model_dir="./checkpoints/")

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    classifier.train(input_fn=lambda: train_input_model(), steps=100, hooks=[logging_hook])

    evaluation = classifier.evaluate(input_fn=lambda: test_input_model())

    print(evaluation)


if __name__ == '__main__':
    main()

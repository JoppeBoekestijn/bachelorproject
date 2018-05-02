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
import time
from datetime import timedelta
tf.logging.set_verbosity(tf.logging.INFO)

# Datapath
data_path = 'dataset_tropic'
filenames = glob.glob(data_path + '/*.tfrecord')

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
img_size_flat = img_size * img_size
num_channels = 3
img_shape = [img_size, img_size, num_channels]
num_classes = 10
test_batch_size = 256

# Running CNN
# batch_size = 32
total_iterations = 0
train_batch_size = 64


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
    weights = new_weights(shape)

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
    # num_featerino = 16 * 16 * 36
    # print("num_featerinos: {}".format(num_featerino))
    num_features = layer_shape[1:4].num_elements()
    # print("num_features: {}".format(num_features))

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

    features = tf.parse_single_example(serialized, features=features)

    # image_raw = features['image/encoded']
    # image = tf.decode_raw(image_raw, tf.uint8)
    # image = tf.cast(image, tf.float32)

    # The labels numbered from 1 to 10 representing the different classes
    label = features['image/class/label'] - 1
    # print('Label shape: {}'.format(label))
    # Names of the classes in the dataset
    # labelText = features['image/class/text']

    image = tf.image.decode_and_crop_jpeg(
        contents=features['image/encoded'],
        channels=num_channels,
        crop_window=[0, 0, img_size, img_size])
    # image = tf.decode_raw(features['image/encoded'], tf.float32)
    # print("image type: {}".format(image.shape))
    # image = tf.cast(image, tf.float32)
    image = (tf.image.convert_image_dtype(image, dtype=tf.float32)) * 255

    # Resize image to (64, 64, 3)
    #image = tf.image.resize_image_with_crop_or_pad(image, img_size, img_size)
    # print("read_and_decode image shape1: {}".format(image.shape))
    # print("read_and_decode image1: {}".format(image))
    # image = flatten_layer(image)
    # print("read_and_decode image shape2: {}".format(image.shape))
    # print("read_and_decode image2: {}".format(image))
    return image, label


def inputs(batch_size, num_epochs):
    """

    :param batch_size:
    :param num_epochs:
    :return:
    """
    if not num_epochs:
        num_epochs = None

    filename_queue = tf.train.string_input_producer(filenames, num_epochs)
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)

    image, label = read_and_decode(serialized)

    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        capacity=1000 + 3 * batch_size,
        num_threads=2,
        min_after_dequeue=1000)

    return image_batch, label_batch


def printing_tfrecords():
    """
    Iterate through filenames and print result
    """
    for example in tf.python_io.tf_record_iterator(filenames):
        result = tf.train.Example.FromString(example)
        print(result)


def plot_images():
    """
    Plots and displays an image
    """
    image, label = inputs(10, num_epochs=None)

    sess = tf.Session()

    with sess.as_default():
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(1):
            img, labeltext = sess.run([image, label])
            img = img.astype(np.uint8)
            for j in range(1):
                plt.imshow(img[j])
                plt.title(labeltext[j])
                plt.show()

        # Stop the threads
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)


def input_fn(train, batch_size=64, buffer_size=10000):
    """

    :param train:
    :param batch_size:
    :param buffer_size:
    :return:
    """
    dataset = tf.data.TFRecordDataset(filenames=filenames)

    dataset = dataset.map(read_and_decode)
    # print("dataset: {}".format(dataset))

    if train:
        dataset = dataset.shuffle(buffer_size=buffer_size)
        num_repeat = None
    else:
        num_repeat = 1

    dataset = dataset.repeat(num_repeat)

    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()

    images_batch, labels_batch = iterator.get_next()
    print("images_batch shape: {}".format(images_batch))
    print("labels_batch shape: {}".format(labels_batch))

    return images_batch, labels_batch


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


# def optimize(num_iterations):
#     """
#
#     :param num_iterations:
#     """
#     global total_iterations
#
#     start_time = time.time()
#
#     # filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
#     # reader = tf.TFRecordReader()
#     # _, serialized = reader.read(filename_queue)
#
#     for i in range(total_iterations,
#                    total_iterations + num_iterations):
#
#         print("ollah")
#         images_batch, labels_batch = train_input_model()
#         # images, labels = read_and_decode(serialized)
#         # images_batch, labels_batch = train_input_model()
#         print("Images batch shape: {}".format(images_batch))
#         print("Labels batch shape: {}".format(labels_batch))
#         # print("Images shape: {}".format(images.shape))
#         #images_x = tf.reshape(images, [num_classes, img_size_flat])
#         # images, labels = sess.run([images_batch, labels_batch])
#         # print("Images shape: {}".format(images))
#         # print("Labels shape: {}".format(labels))
#         feed_dict_train = {x: images_batch,
#                            y_true: labels_batch}
#
#         sess.run(optimizer, feed_dict=feed_dict_train)
#         print("ello")
#
#         if i % 100 == 0:
#             acc = sess.run(accuracy, feed_dict=feed_dict_train)
#
#             msg = "Optimization Iteration: {0:>3}, Training Accuracy {1:>6.1%}"
#
#             print(msg.format(i + 1, acc))
#
#     total_iterations += num_iterations
#
#     end_time = time.time()
#
#     time_dif = end_time - start_time
#
#     print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


# def print_test_accuracy():
#     """
#     Print the accuracy on the test data
#     """
#     num_images = 0
#     dataset = tf.data.TFRecordDataset(filenames=filenames)
#     for fn in filenames:
#         for record in tf.python_io.tf_record_iterator(fn):
#             num_images += 1
#
#     print(num_images)
#     cls_pred = np.zeros(shape=num_images, dtype=np.int)
#
#     i = 0
#     while i < num_images:
#         j = min(i + test_batch_size, num_images)
#         iterator = dataset.make_one_shot_iterator()
#
#         images_batch, labeltext_batch = iterator.get_next()
#
#         feed_dict_train = {x: images_batch,
#                            y_true: labeltext_batch}
#
#         cls_pred[i:j] = sess.run(y_pred_cls, feed_dict=feed_dict_train)
#
#         i = j
#
#     cls_true = labeltext_batch
#
#     correct = (cls_pred == cls_true)
#
#     correct_sum = correct.sum()
#
#     acc = float(correct_sum) / num_images
#
#     msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
#     print(msg.format(acc, correct_sum, num_images))


def cnn_model(features, labels, mode):
    # Placeholder variable for input images, shape is set to [None, img_size_flat],
    # which means that it may hold an arbitrary number of images

    # x = tf.placeholder(tf.float32, shape=[None, img_size_flat * num_channels], name='x')

    print("features shape: {}".format(features.shape))
    print("labels shape: {}".format(labels.shape))

    # Reshape placeholder x such that it can be input to convolutional layer,
    # which expects a 4-dimensional tensor

    # x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

    # print("x_image: {}".format(x_image.shape))

    # Placeholder variable for the true labels associated with the images
    # that were input to placeholder x. Shape is [None, num_classes] which
    # means that it may hold an arbitrary number of labels and each label is
    # a vector of length num_classes

    # y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

    # print("y_true label: {}".format(y_true.shape))

    # initialize variable for class, using the position of the max value (argmax)
    # print("label shape: {}".format(labels))
    # y_true_cls = tf.argmax(labels, axis=1)
    # print("label shape: {}".format(labels))

    # Initialize first convolutional layer
    layer_conv1, weights_conv1 = new_conv_layer(input=features,
                                                num_input_channels=num_channels,
                                                filter_size=filter_size1,
                                                num_filters=num_filters1,
                                                use_pooling=True)
    print("layer_conv1: {}".format(layer_conv1))
    # print(layer_conv1.shape)

    # Initialize second convolutional layer
    layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                                                num_input_channels=num_filters1,
                                                filter_size=filter_size2,
                                                num_filters=num_filters2,
                                                use_pooling=True)
    # print(layer_conv2)
    # print(layer_conv2.shape)
    print("layer_conv2: {}".format(layer_conv2))

    layer_flat, num_features = flatten_layer(layer_conv2)
    print("layer_flat: {}".format(layer_flat.shape))
    # print(num_features)
    # print(layer_flat.shape)

    # Fully-connected layer
    layer_fc1 = new_fc_layer(input=layer_flat,
                             num_inputs=num_features,
                             num_outputs=fc_size,
                             use_relu=True)
    print("layer_fc1: {}".format(layer_fc1.shape))

    # Second fully-connected layer
    layer_fc2 = new_fc_layer(input=layer_fc1,
                             num_inputs=fc_size,
                             num_outputs=num_classes,
                             use_relu=False)
    print("layer_fc2: {}".format(layer_fc2.shape))

    # Normalize prediction using softmax
    y_pred = tf.nn.softmax(layer_fc2)

    # Predicted class is index with largest number
    y_pred_cls = tf.argmax(y_pred, axis=1)

    # Transpose axis of the labels

    # labels = tf.transpose(labels, perm=[1, 0])
    print("labels transpose shape: {}".format(labels.shape))

    # Calculate the overall loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                                   labels=labels)
    loss = tf.reduce_mean(cross_entropy)
    #logits = tf.layers.dense(inputs=layer_fc2, units=2)

    # predictions = {
    #     "classes": tf.argmax(logits, axis=1),
    #     "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    # }

    predictions = {
        "classes": y_pred_cls,
        "probabilities": tf.nn.softmax(layer_fc2, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Cost function to be optimized, calculates softmax internally,
    # # so we must use layer_fc2 instead of y_pred
    # onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    # loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2,
    #                                                            labels=y_true_cls)
    # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=layer_fc2)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


    # We want to have the average cost value instead of per each
    # image classifications
    # cost = tf.reduce_mean(cross_entropy)
    #
    # # Optimization method, use AdamOptimizer which is an advanced form
    # # of gradient descent.
    # tf.train.AdamOptimizer(learning_rate=1e-4)
    #
    # correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    #
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #
    # sess = tf.Session()
    # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # sess.run(init_op)


def main():
    """
    Main function to run convolutional neural network
    """
    # Function to plot one image
    # plot_images()

    # optimize(num_iterations=1)
    # print_test_accuracy()
    classifier = tf.estimator.Estimator(model_fn=cnn_model)

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    classifier.train(input_fn=lambda: train_input_model(), steps=1000, hooks=[logging_hook])

    evaluation = classifier.evaluate(input_fn=lambda: test_input_model())

    print(evaluation)


if __name__ == '__main__':
    main()

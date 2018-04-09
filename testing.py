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

# Shuffle batch
batch_size = 10


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

    # Number of features is: img_size * img-width * num_channels
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features


def read_and_decode():
    """
    Extract image and labels from .TFRecords filequeue
    :return: images and labels
    """
    data_path = 'dataset_tropic'
    filenames = glob.glob(data_path + '/*.tfrecord')

    with tf.Session() as sess:
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        feature = {'image/encoded': tf.FixedLenFeature([], tf.string),
                   'image/class/label': tf.FixedLenFeature([], tf.int64),
                   'image/filename': tf.FixedLenFeature([], tf.string),
                   'image/class/text': tf.FixedLenFeature([], tf.string)}

        features = tf.parse_single_example(serialized_example, features=feature)

        filename = features['image/filename']
        labeltext = features['image/class/text']
        image = tf.image.decode_jpeg(features['image/encoded'], channels=num_channels)
        image = (tf.image.convert_image_dtype(image, dtype=tf.float32) * 255)
        # label = tf.cast(features['image/class/label'], tf.int32)

        image = tf.reshape(image, [250, 250, 3])
        image = tf.image.resize_image_with_crop_or_pad(image, img_size, img_size)
        # image = tf.image.resize_images(image, (64, 64))

        # Any preprocessing here:

        image, filename, labeltext = tf.train.shuffle_batch(
            [image, filename, labeltext],
            batch_size=batch_size,
            capacity=30,
            num_threads=1,
            min_after_dequeue=10)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Stop the threads
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
        sess.close()

        # return image, label, filename
        return image, filename, labeltext


def printing_tfrecords(filenames):
    """
    Iterate through filenames and print result
    :param filenames:
    """
    for example in tf.python_io.tf_record_iterator(filenames):
        result = tf.train.Example.FromString(example)
        print(result)


def tropic_inference():
    """

    """
    # Placeholder variable for input images, shape is set to [None, img_size_flat],
    # which means that it may hold an arbitrary number of images
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

    # Reshape placeholder x such that it can be input to convolutional layer,
    # which expects a 4-dimensional tensor
    x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

    # Placeholder variable for the true labels associated with the images
    # that were input to placeholder x. Shape is [None, num_classes] which
    # means that it may hold an arbitrary number of labels and each label is
    # a vector of length num_classes
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

    # initialize variable for class, using the position of the max value (argmax)
    y_true_cls = tf.argmax(y_true, axis=1)

    # Initialize first convolutional layer
    layer_conv1, weights_conv1 = new_conv_layer(input=x_image,
                                                num_input_channels=num_channels,
                                                filter_size=filter_size1,
                                                num_filters=num_filters1,
                                                use_pooling=True)
    print(layer_conv1)
    print(layer_conv1.shape)

    # Initialize second convolutional layer
    layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                                                num_input_channels=num_filters1,
                                                filter_size=filter_size2,
                                                num_filters=num_filters2,
                                                use_pooling=True)
    print(layer_conv2)
    print(layer_conv2.shape)

    layer_flat, num_features = flatten_layer(layer_conv2)
    print(layer_flat.shape)
    print(num_features)

    # Fully-connected layer
    layer_fc1 = new_fc_layer(input=layer_flat,
                             num_inputs=num_features,
                             num_outputs=fc_size,
                             use_relu=True)
    print(layer_fc1.shape)

    # Second fully-connected layer
    layer_fc2 = new_fc_layer(input=layer_fc1,
                             num_inputs=fc_size,
                             num_outputs=num_classes,
                             use_relu=False)
    print(layer_fc2.shape)

    # Normalize prediction using softmax
    y_pred = tf.nn.softmax(layer_fc2)

    # Predicted class is index with largest number
    y_pred_cls = tf.argmax(y_pred, axis=1)



def plot_images(image, filename, labeltext):
    """
    Plots and displays an image
    """
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for batch_index in range(1):
            img, labeltext = sess.run([image, labeltext])
            img = img.astype(np.uint8)
            for j in range(1):
                plt.imshow(img[j])
                plt.title(labeltext[j])
                plt.show()

        # Stop the threads
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
        sess.close()


def main():
    """
    Main function to run convolutional neural network
    """
    # Reads and decodes all tfrecords files and return images and labels
    image, filename, labeltext = read_and_decode()

    # Function to plot one image
    plot_images(image, filename, labeltext)

    tropic_inference()


if __name__ == '__main__':
    main()

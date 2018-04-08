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
img_size = 250
img_size_flat = img_size * img_size
num_channels = 3
img_shape = [img_size, img_size, num_channels]


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
        image = (tf.image.convert_image_dtype(image, dtype=tf.float32) * img_size)
        #label = tf.cast(features['image/class/label'], tf.int32)
        image = tf.reshape(image, img_shape)

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


def convolutional_neural_network():
    """

    """


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


if __name__ == '__main__':
    main()


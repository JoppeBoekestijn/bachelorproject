# matplotlib inline
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import glob
from PIL import Image

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
img_size = 25
img_size_flat = 25 * 25
img_shape = (img_size, img_size)
num_channels = 3

# Number of labels
num_classes = 10


def read_and_decode(filename_queue):
    """
    Extract image and labels from .TFRecords filequeue
    :param filename_queue:
    :return:
    """
    # feature = {'train/encoded': tf.FixedLenFeature([], tf.string),
    #            'train/class/label': tf.FixedLenFeature([], tf.int64)}
    with tf.Session() as sess:
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        feature = {'train/encoded': tf.FixedLenFeature([], tf.string),
                   'train/class/label': tf.FixedLenFeature([], tf.int64),
                   'train/filename': tf.FixedLenFeature([], tf.int64)}

        features = tf.parse_single_example(serialized_example, features=feature)

        # image = tf.decode_raw(features['train/encoded'], tf.float32)
        # label = tf.cast(features['train/class/label'], tf.int32)

        image = tf.image.decode_jpeg(features['train/encoded'], channels=3)
        label = tf.cast(features['train/class/label'], tf.int64)
        filename = features['train/filename']

        # print(image)
        # print(label)

        image = tf.reshape(image, [25, 25, 3])

        image, label = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1,
                                              min_after_dequeue=10)

        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for batch_index in range(5):
            img, lbl = sess.run([image, label])
            img = img.astype(np.uint8)

            for j in range(6):
                plt.subplot(2, 3, j + 1)
                plt.imshow(img[j, ...])

        # Stop the threads
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
        sess.close()
        # return image, label, filename

    # images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1,
    #                                         min_after_dequeue=10)


def printing_tfrecords(filenames):
    """
    Iterate through filenames and print result
    :param filenames:
    """
    sess = tf.Session()
    record_iterator = tf.python_io.tf_record_iterator(filenames)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        height = int(example.features.feature['image/height'].int64_list.value[0])
        width = int(example.features.feature['image/width'].int64_list.value[0])
        img_string = (example.features.feature['image/encoded'].bytes_list.value[0])
        img_1d = np.fromstring(img_string, dtype=np.uint8)
        # im = Image.fromarray(img_1d).convert('RGB')
        print(img_1d.shape)
        print(img_1d.eval())
        img_3d = tf.image.decode_jpeg(img_string, channels=3)
        img_3d = tf.reshape(img_3d, [250, 250, 3])

        img_3d = tf.cast(img_3d, tf.uint8)

        # print(width)
        # print(img_3d.shape)
        # img_3d = img_3d.astype(np.uint8)
        # for x in range(1):
        #     plt.imshow(img_3d)
        #     plt.show()
        # for x in range(1):
        #     plt.imshow(img_1d)
        #     plt.show()

    sess.close()

    # for example in tf.python_io.tf_record_iterator(filenames):
    #     result = tf.train.Example.ParseFromString(example)
    #     height = int(example.features.feature['image/height'].int64_list.value[0])
    #     width = int(example.features.feature['image/width'].int64_list.value[0])
    #     img_string = int(example.features.feature['image/encoded'].bytes_list.value[0])
    #     img_1d = np.fromstring(img_string, dtype=np.float32)
    #     print(img_1d.shape)


def convolutional_neural_network():
    """

    """


def plot_images(image, label, label_pred=None):
    """

    :param image:
    :param label:
    """
    # sess = tf.Session()

    # castedImage = tf.cast(image, tf.uint8)
    # png_data = tf.image.encode_png(tf.cast(tf.reshape(image, [25, 25, 3]), tf.uint8))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print("Hello")
        # tf.cast(image, tf.int64)
        # print(image)
        image_tensor = sess.run([image])
        print(image_tensor)
        im = Image.fromarray(image_tensor).convert('RGB')
        print(im)

        coord.request_stop()
        coord.join(threads)

        # print(type(png_data))
        # sess.run(tf.global_variables_initializer())
        # png = sess.run(png_data)
        # print(png)
        # print("Hello")

    # sess.close()

    # assert len(png) == len(label_pred) == 9
    #
    # fig, axes = plt.subplots(3, 3)
    # fig.subplots_adjust(hspace=0.3, wspace=0.3)
    #
    # for i, ax in enumerate(axes.flat):
    #     print("yello")
    #     ax.imshow(image[i].reshape(img_shape, cmap='binary'))
    #
    #     # Show true and predicted classes
    #     if label_pred is None:
    #         xlabel = "True: {0}".format(label[i])
    #     else:
    #         xlabel = "True: {0}, Pred: {1}".format(label[i], label_pred[i])
    #
    #     # Show the class as labels on the x-axis
    #     ax.set_xlabel(xlabel)
    #
    #     # Remove ticks from axis
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #
    # plt.show()
    # sess.close()


def main():
    """
    Main function to run convolutional neural network
    """
    data_path = 'dataset_tropic'
    filenames = glob.glob(data_path + '/*.tfrecord')
    dataset = tf.data.TFRecordDataset(filenames)

    # printing_tfrecords('dataset_tropic/train-00000-of-00002.tfrecord')
    # image, label, filename = read_and_decode(filename_queue)
    with tf.Session() as sess:
        filename_queue = tf.train.string_input_producer(filenames,num_epochs=None)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        feature = {'image/encoded': tf.FixedLenFeature([], tf.string),
                   'image/class/label': tf.FixedLenFeature([], tf.int64),
                   'image/filename': tf.FixedLenFeature([], tf.string)}

        features = tf.parse_single_example(serialized_example, features=feature)

        # image = tf.decode_raw(features['train/encoded'], tf.float32)
        # label = tf.cast(features['train/class/label'], tf.int32)
        # image = tf.decode_raw(features['train/encoded'], tf.uint8)
        filename = features['image/filename']
        image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
        image = (tf.image.convert_image_dtype(image, dtype=tf.float32) * 255)
        label = tf.cast(features['image/class/label'], tf.int32)
        image = tf.reshape(image, [250, 250, 3])

        image = tf.train.shuffle_batch(
            [image],
            batch_size=10,
            capacity=30,
            num_threads=1,
            min_after_dequeue=10)


        # print(image)
        # print(label)
        # print(image)
        # print(image.shape)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for batch_index in range(1):
            img, lbl = sess.run([image, label])
            # print(img.shape)
            # print(img)
            img = img.astype(np.uint8)
            #img = img[0, :, :, :]
            print(img.shape)

            for j in range(1):
                plt.imshow(img[j])
                plt.show()

        # Stop the threads
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
        sess.close()
        # return image, label, filename

    # read_and_decode(filename_queue)

    # with tf.Session() as sess:
    #     jpeg_bin = sess.run(image)
    #     print('sess.run: ' + jpeg_bin)
    #     jpeg_str = io.StringIO.io.StringIO(jpeg_bin)
    #     jpeg_image = Image.open(jpeg_str)
    #     plt.imshow(jpeg_image)
    #     print("hello")

    # plot_images(image, label, None)

    # print(image[0].eval(sess))
    # img, lbl = sess.run([image, label])
    # img = img.astype(np.uint8)
    # plt.imshow(img)
    #
    # plt.show()


if __name__ == '__main__':
    main()

# images, labels = tf.train.shuffle_batch([image, label]
# , batch_size=1, capacity=30, num_threads=1, min_after_dequeue=10)

# # Initialize all global and local variables
# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
# sess.run(init_op)
# # Create a coordinator and run all QueueRunner objects
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(coord=coord)
# for batch_index in range(1):
#     img, lbl = sess.run([images, labels])
#     img = img.astype(np.uint8)
#     plt.imshow(img[0])
#     plt.show()
# # Stop the threads
# coord.request_stop()

# Wait for threads to stop
# coord.join(threads)
# sess.close()

# image = tf.decode_raw(features['train/encoded', tf.float32])
# label = tf.cast(features['train/class/label', tf.int32])
# image = tf.reshape(image, [250, 250, 3])
#
# images, labels = tf.train.shuffle_batch([image, label])
#
# print(image)
# print(label)

# filenames = tf.placeholder(tf.string, shape=[None])
# dataset = tf.data.TFRecordDataset(filenames)
#
# iterator = dataset.make_initializable_iterator()
#
# training_filenames = ["dataset_tropic/train-00000-of-00002.tfrecord", "dataset_tropic/train-00001-of-00002.tfrecord"]
# sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

# print(dataset)

# dataset = tf.data.Dataset.range(100)
# iterator = dataset.make_one_shot_iterator()
# next_element = iterator.get_next()
#
#
# for i in range(100):
#     value = sess.run(next_element)
#     assert i == value
#
# print(value)

#################################################################

# max_value = tf.placeholder(tf.int64, [])
# dataset = tf.data.Dataset.range(max_value)
# iterator = dataset.make_initializable_iterator()
# next_element = iterator.get_next()

# Initialize an iterator over a dataset with 10 elements.
# sess.run(iterator.initializer, {max_value: 10})
# for i in range(10):
#     value = sess.run(next_element)
#     assert i == value
#     print(value)

# Initialize an iterator over a dataset with 100 elements.
# sess.run(iterator.initializer, {max_value: 100})
# for i in range(100):
#     value = sess.run(next_element)
#     assert i == value
#     print(value)

#################################################################

# sess.run might run out of bounds, therefore it is common
# to wrap the training loop in a try-except block.
#
# see.run(iterator.initializer)
# while True:
#     try:
#         sess.run(result)
#     except tf.errors.OutOfRangeError:
#         break

#################################################################

# Decoding image data and resizing it
# def _parse_function(filename, label):
#     image_string = tf.read_file(filename)
#     image_decoded = tf.image.decode_image(image_string)
#     return image_decoded, label
#
#
# filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg"])
#
# labels = tf.constant([0, 37])
#
# dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
# dataset = dataset.map(_parse_function)

#matplotlib inline
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import glob


def read_and_decode(filename_queue):
    feature = {'train/encoded': tf.FixedLenFeature([], tf.string),
               'train/class/label': tf.FixedLenFeature([], tf.int64)}

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features=feature)

    image = tf.decode_raw(features['train/encoded'], tf.float32)
    label = tf.cast(features['train/class/label'], tf.int32)

    image = tf.reshape(image, [25, 25, 3])

    # images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1,
    #                                         min_after_dequeue=10)

    return image, label


def printing_tfrecords(filenames):
    for example in tf.python_io.tf_record_iterator(filenames):
        result = tf.train.Example.FromString(example)
        print(result)


# def plot_images(images, labels):
#


def convolutional_neural_network():
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
    img_size_flat = 250*250
    img_shape = (img_size, img_size)
    num_channels = 3

    # Number of labels
    num_classes = 10

def main():
    sess = tf.Session()

    data_path = 'dataset_tropic'
    filenames = glob.glob(data_path + '/*.tfrecord')
    dataset = tf.data.TFRecordDataset(filenames)

    # printing_tfrecords('dataset_tropic/train-00000-of-00002.tfrecord')

    filename_queue = tf.train.string_input_producer(filenames)
    image, label = read_and_decode(filename_queue)

    print(image[0].eval(session=sess))
    img, lbl = sess.run([image, label])
    img = img.astype(np.uint8)
    plt.imshow(img)

    plt.show()

    sess.close()


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

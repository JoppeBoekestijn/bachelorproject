import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os


tf.reset_default_graph()
sess = tf.Session()


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string)
    # Resizes images to a more suitable format. Might be necessary eventually
    # image_resized = tf.image.resize_images(image_decoded, [28, 28])
    return image_decoded, label


dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/dataset_tropic/train"
print(filename)
filenames = tf.constant(["dataset_tropic/train/01_Ashoka/p01_002.png", "dataset_tropic/train/01_Ashoka/p01_003.png"])
labels = tf.constant([0, 1])

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)

print(dataset)


# filenames = ["dataset_tropic_train_0-of-2.tfrecord"]
# "dataset_tropic_train_1-of-2.tfrecord",
# "dataset_tropic_validation_0-of-2.tfrecord",
# "dataset_tropic_validation_1-of-2.tfrecord"]


# data = tf.data.TFRecordDataset(filenames)
# data = data.map(...)
# data = data.repeat()
# data = data.batch(32)
# iterator = data.make_initializable_iterator()
#
# training_filenames = ["dataset_tropic_train_0-of-2.tfrecord"]
# sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
# data = data.repeat(1000)

# for example in tf.python_io.tf_record_iterator(
#         "/Users/joppeboekestijn/Document/create_tfrecords/dataset_tropic/dataset_tropic_train_0-of-2.tfrecord"):
#     result = tf.train.Example.FromString(example)
#     print(result.features.feature['label'].int64_list.value)
#     print(result.features.feature['text_label'].bytes_list.value)

# filenames = "/Users/joppeboekestijn/Document/dataset_tropic/train/01_Ashokap/01_002.png"
# image_content = tf.read_file(filenames)
# image = tf.image.decode_image(image_content, channels=1)
# sess = tf.Session()
# with sess.as_default():
#     sess.run(image)
#     print(image.shape)
#     print(image.eval().shape)

# print(data)
# print("Size of:")
# print("- Training-set:\t\t{}".format(len(data.train.labels)))
# # print("- Test-set:\t\t{}".format(len(data.validation.labels)))
# print("- Validation-set:\t{}".format(len(data.validation.labels)))

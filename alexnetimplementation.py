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

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

# Datapath
data_path = 'dataset_tropic'
train_files = glob.glob(data_path + '/' + 'train-*')
evaluate_files = glob.glob(data_path + '/' + 'validation-*')

# Data dimensions
num_channels = 3
img_size = 224


def read_and_decode(serialized):
    """

    :param serialized:
    :return:
    """
    features = {'image/encoded': tf.FixedLenFeature([], tf.string),
                'image/class/text': tf.FixedLenFeature([], tf.string),
                'image/class/label': tf.FixedLenFeature([], tf.int64)}

    features = tf.parse_single_example(serialized, features=features)

    # The labels numbered from 1 to 10 representing the different classes
    label = tf.cast(tf.reshape(features['image/class/label'], shape=[]), dtype=tf.int32)

    image = tf.image.decode_and_crop_jpeg(
        contents=features['image/encoded'],
        channels=num_channels,
        crop_window=[0, 0, img_size, img_size])

    image = (tf.image.convert_image_dtype(image, dtype=tf.int32))

    return image, label


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

    if train:
        dataset = dataset.shuffle(buffer_size=buffer_size)
        num_repeat = None
    else:
        num_repeat = 1

    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_repeat)
    iterator = dataset.make_one_shot_iterator()

    images_batch, labels_batch = iterator.get_next()
    print("images_batch: {}".format(images_batch))
    print("labels_batch: {}".format(labels_batch))

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


def alexnet_v2_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        biases_initializer=tf.constant_initializer(0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d], padding='SAME'):
            with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
                return arg_sc


def alexnet_v2(inputs,
               num_classes=1000,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='alexnet_v2',
               global_pool=False):
    """AlexNet version 2.
  Described in: http://arxiv.org/pdf/1404.5997v2.pdf
  Parameters from:
  github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
  layers-imagenet-1gpu.cfg
  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224 or set
        global_pool=True. To use in fully convolutional mode, set
        spatial_squeeze to false.
        The LRN layers have been removed and change the initializers from
        random_normal_initializer to xavier_initializer.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: the number of predicted classes. If 0 or None, the logits layer
    is omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      logits. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original AlexNet.)
  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the non-dropped-out input to the logits layer (if num_classes is 0
      or None).
    end_points: a dict of tensors with intermediate activations.
  """
    with tf.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=[end_points_collection]):
            net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID',
                              scope='conv1')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
            net = slim.conv2d(net, 192, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
            net = slim.conv2d(net, 384, [3, 3], scope='conv3')
            net = slim.conv2d(net, 384, [3, 3], scope='conv4')
            net = slim.conv2d(net, 256, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')

            # Use conv2d instead of fully_connected layers.
            with slim.arg_scope([slim.conv2d],
                                weights_initializer=trunc_normal(0.005),
                                biases_initializer=tf.constant_initializer(0.1)):
                net = slim.conv2d(net, 4096, [5, 5], padding='VALID',
                                  scope='fc6')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)
                if global_pool:
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                    end_points['global_pool'] = net
                if num_classes:
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='dropout7')
                    net = slim.conv2d(net, num_classes, [1, 1],
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      biases_initializer=tf.zeros_initializer(),
                                      scope='fc8')
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                    end_points[sc.name + '/fc8'] = net
            return net, end_points


alexnet_v2.default_image_size = 224


def main():
    """
    Main function to run convolutional neural network
    """
    # Function to plot one image
    # plot_images()
    inputs = input_fn(True, 32, 10000)

    with slim.arg_scope(alexnet_v2_arg_scope()):
        outputs, end_points = alexnet_v2(inputs, num_classes=10)

    print(outputs)
    print(end_points)
    # classifier = tf.estimator.Estimator(model_fn=cnn_model_test, model_dir="./checkpoints/")
    #
    # tensors_to_log = {"probabilities": "softmax_tensor"}
    # logging_hook = tf.train.LoggingTensorHook(
    #     tensors=tensors_to_log, every_n_iter=100)
    #
    # classifier.train(input_fn=lambda: train_input_model(), steps=100, hooks=[logging_hook])
    #
    # evaluation = classifier.evaluate(input_fn=lambda: test_input_model())
    #
    # print(evaluation)


if __name__ == '__main__':
    main()

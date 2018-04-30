# def read_and_decode():
#     """
#     Extract image and labels from .TFRecords filequeue
#     :return: images and labels
#     """
#
#     with tf.Session() as sess:
#         filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
#         reader = tf.TFRecordReader()
#         _, serialized_example = reader.read(filename_queue)
#
#         feature = {'image/encoded': tf.FixedLenFeature([], tf.string),
#                    'image/class/label': tf.FixedLenFeature([], tf.int64),
#                    'image/filename': tf.FixedLenFeature([], tf.string),
#                    'image/class/text': tf.FixedLenFeature([], tf.string)}
#
#         features = tf.parse_single_example(serialized_example, features=feature)
#
#         filename = features['image/filename']
#         labeltext = features['image/class/text']
#         image = tf.image.decode_jpeg(features['image/encoded'], channels=num_channels)
#         image = (tf.image.convert_image_dtype(image, dtype=tf.float32) * 255)
#         # label = tf.cast(features['image/class/label'], tf.int32)
#
#         image = tf.reshape(image, [250, 250, 3])
#         image = tf.image.resize_image_with_crop_or_pad(image, img_size, img_size)
#         # image = tf.image.resize_images(image, (64, 64))
#
#         # Any preprocessing here:
#
#         image_batch, filename_batch, labeltext_batch = tf.train.shuffle_batch(
#             [image, filename, labeltext],
#             batch_size=batch_size,
#             capacity=30,
#             num_threads=1,
#             min_after_dequeue=10)
#
        # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # sess.run(init_op)
#
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(coord=coord)
#
#         # Stop the threads
#         coord.request_stop()
#
#         # Wait for threads to stop
#         coord.join(threads)
#         sess.close()
#
#         # return image, label, filename
#         return image_batch, filename_batch, labeltext_batch

# def parse(serialized):
#     """
#
#     :param serialized:
#     :return:
#     """
#     features = {'image/encoded': tf.FixedLenFeature([], tf.string),
#                 'image/class/text': tf.FixedLenFeature([], tf.string)}
#
#     # Parse the serialized data so we get a dict with our data.
#     parsed_example = tf.parse_single_example(serialized=serialized,
#                                              features=features)
#
#     # Get the image as raw bytes.
#     image_raw = parsed_example['image/encoded']
#
#     # Decode the raw bytes so it becomes a tensor with type.
#     image = tf.decode_raw(image_raw, tf.uint8)
#
#     # Convert to float
#     image = tf.cast(image, tf.float32)
#
#     # Get the label associated with the image.
#     label = parsed_example['image/class/text']
#
#     return image, label
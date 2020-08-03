# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Converts Image classification datasets to TFRecords of TF-Example protos.

This module reads the image files data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf
from datasets import dataset_utils


# Seed for repeatability.
_RANDOM_SEED = 0

class ImageReader(object):
	"""Helper class that provides TensorFlow image coding utilities."""

	def __init__(self):
		# Initializes function that decodes RGB JPEG data.
		self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
		self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

	def read_image_dims(self, sess, image_data):
		image = self.decode_jpeg(sess, image_data)
		return image.shape[0], image.shape[1]

	def decode_jpeg(self, sess, image_data):
		image = sess.run(self._decode_jpeg,
						feed_dict={self._decode_jpeg_data: image_data})
		assert len(image.shape) == 3
		assert image.shape[2] == 3
		return image


def _get_filenames_and_classes(dataset_dir, dataset_name):
	"""Returns a list of filenames and inferred class names.

	Args:
		dataset_dir: A directory containing a set of subdirectories representing
		class names. Each subdirectory should contain PNG or JPG encoded images.

	Returns:
		A list of image file paths, relative to `dataset_dir` and the list of
		subdirectories, representing class names.
	"""
	dataset_path = os.path.join(dataset_dir, dataset_name)
	directories = []
	class_names = []
	for category in os.listdir(dataset_path):
		path = os.path.join(dataset_path, category)
		if os.path.isdir(path):
			directories.append(path)
			class_names.append(category)

	photo_filenames = []
	for directory in directories:
		for filename in os.listdir(directory):
			path = os.path.join(directory, filename)
			photo_filenames.append(path)
	
	return photo_filenames, sorted(class_names)


def _get_dataset_filename(dataset_dir, dataset_name, split_name, num_shards, shard_id):
	output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (dataset_name,
		split_name, shard_id, num_shards)
	return os.path.join(dataset_dir, output_filename)


def _convert_dataset(dataset_dir, dataset_name, split_name, filenames, class_names_to_ids, num_shards):
	"""Converts the given filenames to a TFRecord dataset.

	Args:
		split_name: The name of the dataset, either 'train' or 'validation'.
		filenames: A list of absolute paths to png or jpg images.
		class_names_to_ids: A dictionary from class names (strings) to ids
		(integers).
		dataset_dir: The directory where the converted datasets are stored.
	"""
	assert split_name in ['train', 'validation']

	num_per_shard = int(math.ceil(len(filenames) / float(num_shards)))

	with tf.Graph().as_default():
		image_reader = ImageReader()

		with tf.Session('') as sess:

			for shard_id in range(num_shards):
				output_filename = _get_dataset_filename(
					dataset_dir, dataset_name, split_name, num_shards, shard_id)

				with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
					start_ndx = shard_id * num_per_shard
					end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
					for i in range(start_ndx, end_ndx):
						sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
							i+1, len(filenames), shard_id))
						sys.stdout.flush()

						# Read the filename:
						image_data = tf.gfile.GFile(filenames[i], 'rb').read()
						height, width = image_reader.read_image_dims(sess, image_data)

						class_name = os.path.basename(os.path.dirname(filenames[i]))
						class_id = class_names_to_ids[class_name]

						example = dataset_utils.image_to_tfexample(
							image_data, b'jpg', height, width, class_id)
						tfrecord_writer.write(example.SerializeToString())

	sys.stdout.write('\n')
	sys.stdout.flush()


def _dataset_exists(dataset_dir, dataset_name, num_shards):
	for split_name in ['train', 'validation']:
		for shard_id in range(num_shards):
			output_filename = _get_dataset_filename(
			dataset_dir, dataset_name, split_name, num_shards, shard_id)
			if not tf.gfile.Exists(output_filename):
				return False
	return True


def run(dataset_dir, dataset_name, num_shards, val_split=0.2):
	"""Runs the download and conversion operation.

	Args:
		dataset_dir: The dataset directory where the dataset is stored.
	"""
	if not tf.gfile.Exists(dataset_dir):
		tf.gfile.MakeDirs(dataset_dir)

	if _dataset_exists(dataset_dir, dataset_name, num_shards):
		print('Dataset files already exist. Exiting without re-creating them.')
		return

	photo_filenames, class_names = _get_filenames_and_classes(dataset_dir, dataset_name)

	class_names_to_ids = dict(
		list(zip(class_names, list(range(len(class_names))))))

	# Divide into train and test:
	random.seed(_RANDOM_SEED)
	random.shuffle(photo_filenames)

	num_validation = int(len(photo_filenames) * val_split)
	num_train = len(photo_filenames) -  num_validation

	training_filenames = photo_filenames[num_validation:]
	validation_filenames = photo_filenames[:num_validation]

	# First, convert the training and validation sets.
	_convert_dataset(dataset_dir, dataset_name,'train', training_filenames, class_names_to_ids,
					num_shards)
	_convert_dataset(dataset_dir, dataset_name, 'validation', validation_filenames, class_names_to_ids,
					num_shards)

	# Finally, write the labels file:
	labels_to_class_names = dict(
		list(zip(list(range(len(class_names))), class_names)))
	dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

	# Finally, write the datasets splits file
	dataset_split_dict = dict(train=num_train, validation=num_validation)
	dataset_utils.write_dataset_split_file(dataset_split_dict, dataset_dir)
	#_clean_up_temporary_files(dataset_dir)
	print('\nFinished converting the %s dataset!' %dataset_name)

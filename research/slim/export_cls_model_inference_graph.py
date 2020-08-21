"""Tool to export a model for inference.

Outputs inference graph, asscociated checkpoint files, a frozen inference
graph and a SavedModel (https://tensorflow.github.io/serving_basic.html).

The inference graph contains one of three input nodes depending on the user
specified option.
    * 'image_tensor': Accepts a uint8 4-D tensor of shape [None, None, None, 3]
    * 'encoded_image_string_tensor': Accepts a 1-D string tensor of shape 
        [None] containg encoded PNG or JPEG images.
    * 'tf_example': Accepts a 1-D string tensor of shape [None] containing
        serialized TFExample protos.
        
and the following output nodes returned by the model.postprocess(..):
    * 'classes': Outputs float32 tensors of the form [batch_size] containing
        the classes for the predictions.
        
Example Usage:
---------------
python/python3 export_inference_graph \
    --input_type image_tensor \
    --trained_checkpoint_prefix path/to/model.ckpt \
    --output_directory path/to/exported_model_directory
    
The exported output would be in the directory
path/to/exported_model_directory (which is created if it does not exist)
with contents:
    - model.ckpt.data-00000-of-00001
    - model.ckpt.info
    - model.ckpt.meta
    - frozen_inference_graph.pb
    + saved_model (a directory)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import tensorflow.compat.v1 as tf
from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.python.platform import gfile
from datasets import dataset_factory
from nets import nets_factory
import exporter


tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to save.')
tf.app.flags.DEFINE_boolean(
    'is_training', False,
    'Whether to save out a training-focused version of the model.')
tf.app.flags.DEFINE_integer(
    'image_size', None,
    'The image size to use, otherwise use the model default_image_size.')
tf.app.flags.DEFINE_integer(
    'batch_size', None,
    'Batch size for the exported model. Defaulted to "None" so batch size can '
    'be specified at model runtime.')
tf.app.flags.DEFINE_string('dataset_name', 'imagenet',
                           'The name of the dataset to use with the model.')
tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')
tf.app.flags.DEFINE_string(
    'dataset_dir', '', 'Directory to save intermediate dataset files to')
tf.flags.DEFINE_string('input_type', 'image_tensor', 'Type of input node. Can '
                    "be one of ['image_tensor', 'encoded_image_string_tensor'"
                    ", 'tf_example']")
tf.flags.DEFINE_string('trained_checkpoint_prefix', None,
                    'Path to trained checkpoint, typically of the form '
                    'path/to/model.ckpt')
tf.flags.DEFINE_string('output_directory', None, 'Path to write outputs')


FLAGS = tf.app.flags.FLAGS

def main(_):
    tf.app.flags.mark_flag_as_required('trained_checkpoint_prefix')
    tf.app.flags.mark_flag_as_required('output_directory')
    tf.logging.set_verbosity(tf.logging.INFO)

    dataset = dataset_factory.get_dataset(FLAGS.dataset_name, 'train',
                                        FLAGS.dataset_dir)
    network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            is_training=FLAGS.is_training)
    image_size = FLAGS.image_size or network_fn.default_image_size

    input_shape = [FLAGS.batch_size, image_size, image_size, 3]

    exporter.export_inference_graph(FLAGS.input_type,
                                    network_fn,
                                    FLAGS.trained_checkpoint_prefix,
                                    FLAGS.output_directory,
                                    input_shape)
    
if __name__ == '__main__':
    tf.app.run()


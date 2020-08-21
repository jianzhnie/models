from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import tensorflow.compat.v1 as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.contrib import quantize as contrib_quantize

from tensorflow.python.platform import gfile
from datasets import dataset_factory
from nets import nets_factory

"""
1、input_graph：（必选）模型文件，可以是二进制的pb文件，或文本的meta文件，用input_binary来指定区分（见下面说明） 
2、input_saver：（可选）Saver解析器。保存模型和权限时，Saver也可以自身序列化保存，以便在加载时应用合适的版本。主要用于版本不兼容时使用。可以为空，为空时用当前版本的Saver。 
3、input_binary：（可选）配合input_graph用，为true时，input_graph为二进制，为false时，input_graph为文件。默认False 
4、input_checkpoint：（必选）检查点数据文件。训练时，给Saver用于保存权重、偏置等变量值。这时用于模型恢复变量值。 
5、output_node_names：（必选）输出节点的名字，有多个时用逗号分开。用于指定输出节点，将没有在输出线上的其它节点剔除。 
6、restore_op_name：（可选）从模型恢复节点的名字。升级版中已弃用。默认：save/restore_all 
7、filename_tensor_name：（可选）已弃用。默认：save/Const:0 
8、output_graph：（必选）用来保存整合后的模型输出文件。 
9、clear_devices：（可选），默认True。指定是否清除训练时节点指定的运算设备（如cpu、gpu、tpu。cpu是默认） 
10、initializer_nodes：（可选）默认空。权限加载后，可通过此参数来指定需要初始化的节点，用逗号分隔多个节点名字。 
11、variable_names_blacklist：（可先）默认空。变量黑名单，用于指定不用恢复值的变量，用逗号分隔多个变量名字。 
"""

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

tf.app.flags.DEFINE_bool(
    'quantize', False, 'whether to use quantized graph or not.')

tf.app.flags.DEFINE_bool('write_text_graphdef', True,
                         'Whether to write a text version of graphdef.')

tf.app.flags.DEFINE_bool('use_grayscale', False,
                         'Whether to convert input images to grayscale.')

tf.app.flags.DEFINE_string(
    'output_dir', '', 'Where to save the resulting file to.')

tf.app.flags.DEFINE_string(
    'ckpt_file', '', 'The checkpoint file uses for convert.')

tf.app.flags.DEFINE_string(
    'output_prototxt_file', 'export_model.prototxt', 'Where to save the resulting file to.')

tf.app.flags.DEFINE_string(
    'output_pb_file', 'export_model.pb', 'Where to save the resulting file to.')

FLAGS = tf.app.flags.FLAGS

FLAGS.output_prototxt_file = os.path.join(FLAGS.output_dir, FLAGS.output_prototxt_file)
FLAGS.output_pb_file = os.path.join(FLAGS.output_dir, FLAGS.output_pb_file)


output_node_names = dict(resnet_v1_50='resnet_v1_50/predictions/Reshape_1', 
                        inception_v3='InceptionV3/Predictions/Reshape_1',
                        vgg_16='vgg_16/Predictions/Reshape_1')

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default() as graph:
        dataset = dataset_factory.get_dataset(FLAGS.dataset_name, 'train',
                                          FLAGS.dataset_dir)
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            is_training=FLAGS.is_training)
        image_size = FLAGS.image_size or network_fn.default_image_size
        num_channels = 1 if FLAGS.use_grayscale else 3

        input_shape = [FLAGS.batch_size, image_size, image_size, num_channels]
        placeholder = tf.placeholder(name='input', dtype=tf.float32,
                                    shape=input_shape)
        network_fn(placeholder)

        if FLAGS.quantize:
            contrib_quantize.create_eval_graph()

        graph_def = graph.as_graph_def()
        if FLAGS.write_text_graphdef:
            tf.io.write_graph(
                graph_def,
                os.path.dirname(FLAGS.output_prototxt_file),
                os.path.basename(FLAGS.output_prototxt_file),
                as_text=True)
        else:
            with gfile.GFile(FLAGS.output_prototxt_file, 'wb') as f:
                f.write(graph_def.SerializeToString())

    freeze_graph.freeze_graph(input_graph=FLAGS.output_prototxt_file,
                            input_saver='',
                            input_binary=False,
                            input_checkpoint=FLAGS.ckpt_file,
                            output_node_names=output_node_names[FLAGS.model_name], # need to modify across different network
                            restore_op_name='save/restore_all',
                            filename_tensor_name='save/Const:0',
                            output_graph=FLAGS.output_pb_file,
                            clear_devices=True,
                            initializer_nodes='',
                            variable_names_blacklist='')


if __name__ == '__main__':
    tf.app.run()
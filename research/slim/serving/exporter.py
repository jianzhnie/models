import os
import tensorflow.compat.v1 as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.contrib import quantize as contrib_quantize

from tensorflow.python.platform import gfile
from datasets import dataset_factory
from nets import nets_factory


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


    with tf.Session(graph=tf.Graph()) as sess:
        input_tensor = tf.placeholder(name='input', dtype=tf.float32,
                                    shape=input_shape)
        # perform inference on the input image
        logits_tf = network_fn(input_tensor)
        # extract the segmentation mask
        predictions_tf = tf.argmax(logits_tf, axis=3)
        
    
        # specify the directory where the pre-trained model weights are stored
        pre_trained_model_dir = os.path.join(, model_name, "train")

        saver = tf.train.Saver()

        # Restore variables from disk.
        saver.restore(sess, os.path.join(pre_trained_model_dir, "model.ckpt"))
        print("Model", model_name, "restored.")

        # Create SavedModelBuilder class
        # defines where the model will be exported
        export_path_base = FLAGS.export_model_dir
        export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(FLAGS.model_version)))
        print('Exporting trained model to', export_path)
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
         
        

def export_model(session, m):
   #只需要修改这一段，定义输入输出，其他保持默认即可
    model_signature = signature_def_utils.build_signature_def(
        inputs={"input": utils.build_tensor_info(m.a)},
        outputs={
            "output": utils.build_tensor_info(m.y)},

        method_name=signature_constants.PREDICT_METHOD_NAME)

    export_path = "pb_model/1"
    if os.path.exists(export_path):
        os.system("rm -rf "+ export_path)
    print("Export the model to {}".format(export_path))

    try:
        legacy_init_op = tf.group(
            tf.tables_initializer(), name='legacy_init_op')
        builder = saved_model_builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            session, [tag_constants.SERVING],
            clear_devices=True,
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    model_signature,
            },
            legacy_init_op=legacy_init_op)

        builder.save()
    except Exception as e:
        print("Fail to export saved model, exception: {}".format(e))


          
def save_model():
    input_shape = [FLAGS.batch_size, image_size, image_size, num_channels]
    placeholder = tf.placeholder(name='input', dtype=tf.float32,
                                shape=input_shape)
    output = tf.placeholder(name='output', dtype=tf.float32)
    
    with tf.Session() as sess:
        model_path = './model/saved_model_builder/1/'
        builder = tf.saved_model.builder.SavedModelBuilder(model_path)        
        inputs = {'input': tf.saved_model.utils.build_tensor_info(placeholder)}
        outputs = {'output': tf.saved_model.utils.build_tensor_info(output)}        
        method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME        
        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, method_name)        
        builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING], 
                signature_def_map={'predict_signature': prediction_signature})
        builder.save()
 
def main():
    save_model()
 
if __name__ == '__main__':
    main()
# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

r"""Creates and runs TF2 object detection models.

For local training/evaluation run:
PIPELINE_CONFIG_PATH=path/to/pipeline.config
MODEL_DIR=/tmp/model_outputs
NUM_TRAIN_STEPS=10000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python model_main_tf2.py -- \
  --model_dir=$MODEL_DIR --num_train_steps=$NUM_TRAIN_STEPS \
  --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
  --pipeline_config_path=$PIPELINE_CONFIG_PATH \
  --alsologtostderr
"""
import os
import shutil
from absl import flags
import logging
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
from object_detection import model_lib_v2
from google.protobuf import text_format
from exporter_lib_v2 import export_inference_graph
from object_detection.protos import pipeline_pb2


from dataset_tools.create_coco_tf_record_ import _create_tf_record_from_coco_annotations
from dataset_tools.build_config import update_configs

logger = tf.get_logger()
logger.setLevel(logging.INFO)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

## model
#######################
# DATASETS DEFINE #
#######################
flags.DEFINE_boolean(
    'include_masks', False, 'Whether to include instance segmentations masks '
                            '(PNG encoded) in the result. default: False.')
flags.DEFINE_string('train_keypoint_annotations_file', '',
                    'Training annotations JSON file.')
flags.DEFINE_string('val_keypoint_annotations_file', '',
                    'Validation annotations JSON file.')
# DensePose is only available for coco 2014.
flags.DEFINE_string('train_densepose_annotations_file', '',
                    'Training annotations JSON file for DensePose.')
flags.DEFINE_string('val_densepose_annotations_file', '',
                    'Validation annotations JSON file for DensePose.')
# Whether to only produce images/annotations on person class (for keypoint /
# densepose task).
flags.DEFINE_boolean('remove_non_person_annotations', False, 'Whether to '
                                                             'remove all annotations for non-person objects.')
flags.DEFINE_boolean('remove_non_person_images', False, 'Whether to '
                                                        'remove all examples that do not contain a person.')
flags.DEFINE_string('train_input_path', None, 'train_input_path')
flags.DEFINE_string('eval_input_path', None, 'eval_input_path')

## model
#######################
# MODEL DEFINE #
#######################
flags.DEFINE_string('input_type', 'image_tensor', 'Type of input node. Can be '
                                                  'one of [`image_tensor`, `encoded_image_string_tensor`, '
                                                  '`tf_example`, `float_image_tensor`]')
flags.DEFINE_string('data_path', None, 'The directory where the original image dataset files are stored.')
flags.DEFINE_string(
    'output_path', None, 'Directory where checkpoints and event logs are written to.')
flags.DEFINE_string(
    'dataset_name', 'user', 'The name of the dataset to load.')
flags.DEFINE_integer(
    'batch_size', 2, 'The number of samples in each batch.')
flags.DEFINE_integer("num_classes", 90, "Number of training classes.")
flags.DEFINE_string('fine_tune_checkpoint', '/data/premodel/code/object_detection/data/ssd_efficientdet_d0/ckpt-0',
                    'The path to a checkpoint from which to fine-tune.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_string('label_map_path', 'data/mscoco_label_map.pbtxt', 'Path to label map file.')
flags.DEFINE_string('pipeline_config_path',
                    '/data/premodel/code/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_gpu.config',
                    'Path to pipeline config file.')
flags.DEFINE_integer('num_train_steps', 1000, 'Number of train steps.')
flags.DEFINE_bool('eval_on_train_data', False, 'Enable evaluating on train '
                                               'data (only supported in distributed training).')
flags.DEFINE_integer('sample_1_of_n_eval_examples', None, 'Will sample one of '
                                                          'every n eval input examples, where n is provided.')
flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample '
                                                                'one of every n train input examples for evaluation, '
                                                                'where n is provided. This is only used if '
                                                                '`eval_training_data` is True.')
flags.DEFINE_string(
    'checkpoint_path', None, 'Path to directory holding a checkpoint.  If '
                            '`checkpoint_path` is provided, this binary operates in eval-only mode, '
                            'writing resulting metrics to `model_dir`.')

flags.DEFINE_integer('eval_timeout', 60, 'Number of seconds to wait for an'
                                          'evaluation checkpoint before exiting.')

flags.DEFINE_bool('use_tpu', False, 'Whether the job is executing on a TPU.')
flags.DEFINE_string(
    'tpu_name',
    default=None,
    help='Name of the Cloud TPU for Cluster Resolvers.')
flags.DEFINE_integer(
    'num_workers', 1, 'When num_workers > 1, training uses '
                      'MultiWorkerMirroredStrategy. When num_workers = 1 it uses '
                      'MirroredStrategy.')
flags.DEFINE_integer(
    'checkpoint_every_n', 100, 'Integer defining how often we checkpoint.')
flags.DEFINE_boolean('record_summaries', True,
                     ('Whether or not to record summaries during'
                      ' training.'))
flags.DEFINE_string('config_override', '',
                    'pipeline_pb2.TrainEvalPipelineConfig '
                    'text proto to override pipeline_config_path.')
flags.DEFINE_boolean('use_side_inputs', False,
                     'If True, uses side inputs as well as image inputs.')
flags.DEFINE_string('side_input_shapes', '',
                    'If use_side_inputs is True, this explicitly sets '
                    'the shape of the side input tensors to a fixed size. The '
                    'dimensions are to be provided as a comma-separated list '
                    'of integers. A value of -1 can be used for unknown '
                    'dimensions. A `/` denotes a break, starting the shape of '
                    'the next side input tensor. This flag is required if '
                    'using side inputs.')
flags.DEFINE_string('side_input_types', '',
                    'If use_side_inputs is True, this explicitly sets '
                    'the type of the side input tensors. The '
                    'dimensions are to be provided as a comma-separated list '
                    'of types, each of `string`, `integer`, or `float`. '
                    'This flag is required if using side inputs.')
flags.DEFINE_string('side_input_names', '',
                    'If use_side_inputs is True, this explicitly sets '
                    'the names of the side input tensors required by the model '
                    'assuming the names will be a comma-separated list of '
                    'strings. This flag is required if using side inputs.')


FLAGS = flags.FLAGS


def create_label_map(train_annotations_file):
    import json
    data = json.load(open(train_annotations_file, 'r'))
    categories = data['categories']
    FLAGS.num_classes = len(categories)
    FLAGS.label_map_path = os.path.join(FLAGS.output_path, "label.pbtxt")
    with open(FLAGS.label_map_path, "w")as f:
        for categorie in categories:
            f.write('item{\n' + 'name:"' + categorie['name'] + '"\nid:' + str(categorie['id']) + '\ndisplay_name:"' +
                    categorie['name'] + '"\n}')

def creat_tf_record():
    if FLAGS.dataset_name == 'coco':
        create_coco_tf_record()
    else:
        create_custom_tf_record()


def create_custom_tf_record():
    assert os.path.exists(FLAGS.data_path), 'dataset path do not exits.'

    train_output_path = os.path.join(FLAGS.output_path, 'tf_record', FLAGS.dataset_name + '_train.record')
    val_output_path = os.path.join(FLAGS.output_path, 'tf_record', FLAGS.dataset_name + '_val.record')

    train_annotations_file = os.path.join(FLAGS.data_path, "format_coco/annotations/instance.json")

    train_image_dir = os.path.join(FLAGS.data_path, "format_coco/images")

    val_annotations_file = os.path.join(FLAGS.data_path, "format_coco/annotations/instance.json")
    val_image_dir = os.path.join(FLAGS.data_path, "format_coco/images")

    output_tfrecord_dir = os.path.join(FLAGS.output_path, 'tf_record')
    if not tf.gfile.IsDirectory(output_tfrecord_dir):
        tf.gfile.MakeDirs(output_tfrecord_dir)
    

    _create_tf_record_from_coco_annotations(
        train_annotations_file,
        train_image_dir,
        train_output_path,
        FLAGS.include_masks,
        num_shards=1,
        keypoint_annotations_file=FLAGS.train_keypoint_annotations_file,
        densepose_annotations_file=FLAGS.train_densepose_annotations_file,
        remove_non_person_annotations=FLAGS.remove_non_person_annotations,
        remove_non_person_images=FLAGS.remove_non_person_images)

    _create_tf_record_from_coco_annotations(
        val_annotations_file,
        val_image_dir,
        val_output_path,
        FLAGS.include_masks,
        num_shards=1,
        keypoint_annotations_file=FLAGS.val_keypoint_annotations_file,
        densepose_annotations_file=FLAGS.val_densepose_annotations_file,
        remove_non_person_annotations=FLAGS.remove_non_person_annotations,
        remove_non_person_images=FLAGS.remove_non_person_images)

    FLAGS.train_input_path = os.path.join(output_tfrecord_dir, FLAGS.dataset_name + '_train.record-?????-of-00001')
    FLAGS.eval_input_path = os.path.join(output_tfrecord_dir, FLAGS.dataset_name + '_val.record-?????-of-00001')
    create_label_map(train_annotations_file)


def create_coco_tf_record():
    assert os.path.exists(FLAGS.data_path), 'dataset path do not exits.'

    train_annotations_file = os.path.join(FLAGS.data_path , "annotations/instances_val2017.json")
    train_image_dir = os.path.join(FLAGS.data_path , "val2017")

    val_annotations_file = os.path.join(FLAGS.data_path , "annotations/instances_val2017.json")
    val_image_dir = os.path.join(FLAGS.data_path , "val2017")

    testdev_annotations_file = os.path.join(FLAGS.data_path , "annotations/instances_val2017.json")
    test_image_dir = os.path.join(FLAGS.data_path , "val2017/")

    output_tfrecord_dir = os.path.join(FLAGS.output_path, 'tf_record')
    if not tf.gfile.IsDirectory(output_tfrecord_dir):
        tf.gfile.MakeDirs(output_tfrecord_dir)

    train_output_path = os.path.join(FLAGS.output_path, 'tf_record', 'coco_train.record')
    val_output_path = os.path.join(FLAGS.output_path, 'tf_record', 'coco_val.record')
    testdev_output_path = os.path.join(FLAGS.output_path,'tf_record', 'coco_testdev.record')

    _create_tf_record_from_coco_annotations(
        train_annotations_file,
        train_image_dir,
        train_output_path,
        FLAGS.include_masks,
        num_shards=100,
        keypoint_annotations_file=FLAGS.train_keypoint_annotations_file,
        densepose_annotations_file=FLAGS.train_densepose_annotations_file,
        remove_non_person_annotations=FLAGS.remove_non_person_annotations,
        remove_non_person_images=FLAGS.remove_non_person_images)
    _create_tf_record_from_coco_annotations(
        val_annotations_file,
        val_image_dir,
        val_output_path,
        FLAGS.include_masks,
        num_shards=50,
        keypoint_annotations_file=FLAGS.val_keypoint_annotations_file,
        densepose_annotations_file=FLAGS.val_densepose_annotations_file,
        remove_non_person_annotations=FLAGS.remove_non_person_annotations,
        remove_non_person_images=FLAGS.remove_non_person_images)
    _create_tf_record_from_coco_annotations(
        testdev_annotations_file,
        test_image_dir,
        testdev_output_path,
        FLAGS.include_masks,
        num_shards=50)

    FLAGS.train_input_path = os.path.join(output_tfrecord_dir, FLAGS.dataset_name + '_train.record-?????-of-00100')
    FLAGS.eval_input_path = os.path.join(output_tfrecord_dir, FLAGS.dataset_name + '_val.record-?????-of-00050')

    create_label_map(train_annotations_file)


def remove_prev_models():
    if not FLAGS.checkpoint_path:
        # 刪除之前的checkponit文件夹
        if os.path.isdir(FLAGS.output_path):
            files = os.listdir(FLAGS.output_path)
            for f in files:
                if "ckpt" in f or 'check' in f :
                    path = os.path.join(FLAGS.output_path, f)
                    if not os.path.isdir(path):
                        os.remove(os.path.join(FLAGS.output_path, f))

def export_model():
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf2.io.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    text_format.Merge(FLAGS.config_override, pipeline_config)
    output_directory = os.path.join(FLAGS.output_path, 'export')
    export_inference_graph(
        FLAGS.input_type, pipeline_config, FLAGS.output_path,
        output_directory, FLAGS.use_side_inputs, FLAGS.side_input_shapes,
        FLAGS.side_input_types, FLAGS.side_input_names)


def main(unused_argv):
    flags.mark_flag_as_required('data_path')
    flags.mark_flag_as_required('output_path')
    flags.mark_flag_as_required('dataset_name')
    tf2.config.set_soft_device_placement(True)
    print("<<<<<<<<<<<<<\nSTART REMOVE BEFORE WORKDIRS\n<<<<<<<<<<<<<")
    remove_prev_models()
    print("<<<<<<<<<<<<<\nSTART CREATE TFRECORD \n<<<<<<<<<<<<<")
    creat_tf_record()
    print("<<<<<<<<<<<<<\nSTART UPDATE CONFIG \n<<<<<<<<<<<<<")
    update_configs(pipeline_config_path=FLAGS.pipeline_config_path,
                   model_dir=FLAGS.output_path,
                   train_steps=FLAGS.num_train_steps,
                   batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   fine_tune_checkpoint=FLAGS.fine_tune_checkpoint,
                   label_map_path=FLAGS.label_map_path,
                   train_input_path=FLAGS.train_input_path,
                   eval_input_path=FLAGS.eval_input_path,
                   num_classes=FLAGS.num_classes,)
    FLAGS.pipeline_config_path = os.path.join(FLAGS.output_path, 'pipeline.config')

    if FLAGS.checkpoint_path:
        print("<<<<<<<<<<<<<\nSTART EVALUATION  \n<<<<<<<<<<<<<")
        model_lib_v2.eval_continuously(
            pipeline_config_path=FLAGS.pipeline_config_path,
            model_dir=FLAGS.output_path,
            train_steps=FLAGS.num_train_steps,

            sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
            sample_1_of_n_eval_on_train_examples=(
                FLAGS.sample_1_of_n_eval_on_train_examples),
            checkpoint_dir=FLAGS.checkpoint_path,
            wait_interval=30, timeout=FLAGS.eval_timeout)
    else:
        if FLAGS.use_tpu:
            # TPU is automatically inferred if tpu_name is None and
            # we are running under cloud ai-platf2orm.
            resolver = tf2.distribute.cluster_resolver.TPUClusterResolver(
                FLAGS.tpu_name)
            tf2.config.experimental_connect_to_cluster(resolver)
            tf2.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf2.distribute.experimental.TPUStrategy(resolver)
        elif FLAGS.num_workers > 1:
            strategy = tf2.distribute.experimental.MultiWorkerMirroredStrategy()
        else:
            strategy = tf2.compat.v2.distribute.MirroredStrategy()

        with strategy.scope():
            model_lib_v2.train_loop(
                pipeline_config_path=FLAGS.pipeline_config_path,
                model_dir=FLAGS.output_path,
                train_steps=FLAGS.num_train_steps,
                use_tpu=FLAGS.use_tpu,
                checkpoint_every_n=FLAGS.checkpoint_every_n,
                record_summaries=FLAGS.record_summaries)
        #######################
        # Export models #
        #######################
        print("<<<<<<<<<<<<<\nEXPORT MODEL  \n<<<<<<<<<<<<<")
        print(
            f"python /data/premodel/code/object_detection/exporter_main_v2.py --input_type=image_tensor  --pipeline_config_path={FLAGS.pipeline_config_path}  --trained_checkpoint_dir={FLAGS.output_path} --output_directory={os.path.join(FLAGS.output_path, 'export')} --side_input_shapes={FLAGS.side_input_shapes}   --side_input_shapes={FLAGS.side_input_types} --side_input_names={FLAGS.side_input_names}  --use_side_inputs={FLAGS.use_side_inputs}")
        os.system(
            f"python /data/premodel/code/object_detection/exporter_main_v2.py --input_type=image_tensor  --pipeline_config_path={FLAGS.pipeline_config_path}  --trained_checkpoint_dir={FLAGS.output_path} --output_directory={os.path.join(FLAGS.output_path, 'export')} --side_input_shapes={FLAGS.side_input_shapes}   --side_input_shapes={FLAGS.side_input_types} --side_input_names={FLAGS.side_input_names}  --use_side_inputs={FLAGS.use_side_inputs}")





if __name__ == '__main__':
    tf2.compat.v1.app.run()
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
from absl import flags
import logging
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
from object_detection import model_lib_v2
from dataset_tools.create_coco_tf_record_ import _create_tf_record_from_coco_annotations


logger = tf.get_logger()
logger.setLevel(logging.INFO)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


#######################
# Dataset Flags #
#######################
flags1 = tf.app.flags
tf.flags.DEFINE_boolean(
    'include_masks', False, 'Whether to include instance segmentations masks '
    '(PNG encoded) in the result. default: False.')
tf.flags.DEFINE_string('train_image_dir', '', 'Training image directory.')
tf.flags.DEFINE_string('val_image_dir', '', 'Validation image directory.')
tf.flags.DEFINE_string('test_image_dir', '', 'Test image directory.')
tf.flags.DEFINE_string('train_annotations_file', '',
                       'Training annotations JSON file.')
tf.flags.DEFINE_string('val_annotations_file', '',
                       'Validation annotations JSON file.')
tf.flags.DEFINE_string('testdev_annotations_file', '',
                       'Test-dev annotations JSON file.')
tf.flags.DEFINE_string('train_keypoint_annotations_file', '',
                       'Training annotations JSON file.')
tf.flags.DEFINE_string('val_keypoint_annotations_file', '',
                       'Validation annotations JSON file.')
# DensePose is only available for coco 2014.
tf.flags.DEFINE_string('train_densepose_annotations_file', '',
                       'Training annotations JSON file for DensePose.')
tf.flags.DEFINE_string('val_densepose_annotations_file', '',
                       'Validation annotations JSON file for DensePose.')
tf.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')
# Whether to only produce images/annotations on person class (for keypoint /
# densepose task).
tf.flags.DEFINE_boolean('remove_non_person_annotations', False, 'Whether to '
                        'remove all annotations for non-person objects.')
tf.flags.DEFINE_boolean('remove_non_person_images', False, 'Whether to '
                        'remove all examples that do not contain a person.')

## model
#######################
# MODEL DEFINE #
#######################
flags.DEFINE_string('data_path', None, 'The directory where the original image dataset files are stored.')
flags.DEFINE_string(
    'output_path', None, 'Directory where checkpoints and event logs are written to.')
flags.DEFINE_string(
    'dataset_name', 'coco', 'The name of the dataset to load.')
flags.DEFINE_integer(
    'batch_size', 2, 'The number of samples in each batch.')
flags.DEFINE_integer(
    "num_classes", 90,  help="Number of training classes.")
flags.DEFINE_string(
    'fine_tune_checkpoint', 'datasets/work_dirs/checkpoints/efficientnet_b0/ckpt-0',
    'The path to a checkpoint from which to fine-tune.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_string('label_map_path', 'data/mscoco_label_map.pbtxt', 'Path to  label map file.')
flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
                    'file.')
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
    'model_dir', None, 'Path to output model directory '
                                                  'where event and checkpoint files will be written.')
flags.DEFINE_string(
    'checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
                            '`checkpoint_dir` is provided, this binary operates in eval-only mode, '
                            'writing resulting metrics to `model_dir`.')

flags.DEFINE_integer('eval_timeout', 120, 'Number of seconds to wait for an'
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

def create_coco_tf_record():
  assert FLAGS.train_image_dir, '`train_image_dir` missing.'
  assert FLAGS.val_image_dir, '`val_image_dir` missing.'
  assert FLAGS.test_image_dir, '`test_image_dir` missing.'
  assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
  assert FLAGS.val_annotations_file, '`val_annotations_file` missing.'
  assert FLAGS.testdev_annotations_file, '`testdev_annotations_file` missing.'

  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)
  train_output_path = os.path.join(FLAGS.output_dir, 'coco_train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'coco_val.record')
  testdev_output_path = os.path.join(FLAGS.output_dir, 'coco_testdev.record')


  _create_tf_record_from_coco_annotations(
      FLAGS.train_annotations_file,
      FLAGS.train_image_dir,
      train_output_path,
      FLAGS.include_masks,
      num_shards=1,
      keypoint_annotations_file=FLAGS.train_keypoint_annotations_file,
      densepose_annotations_file=FLAGS.train_densepose_annotations_file,
      remove_non_person_annotations=FLAGS.remove_non_person_annotations,
      remove_non_person_images=FLAGS.remove_non_person_images)
  _create_tf_record_from_coco_annotations(
      FLAGS.val_annotations_file,
      FLAGS.val_image_dir,
      val_output_path,
      FLAGS.include_masks,
      num_shards=50,
      keypoint_annotations_file=FLAGS.val_keypoint_annotations_file,
      densepose_annotations_file=FLAGS.val_densepose_annotations_file,
      remove_non_person_annotations=FLAGS.remove_non_person_annotations,
      remove_non_person_images=FLAGS.remove_non_person_images)
  _create_tf_record_from_coco_annotations(
      FLAGS.testdev_annotations_file,
      FLAGS.test_image_dir,
      testdev_output_path,
      FLAGS.include_masks,
      num_shards=50)


def main(unused_argv):
  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('pipeline_config_path')
  tf2.config.set_soft_device_placement(True)

  if FLAGS.checkpoint_dir:
    model_lib_v2.eval_continuously(
        pipeline_config_path=FLAGS.pipeline_config_path,
        model_dir=FLAGS.model_dir,
        train_steps=FLAGS.num_train_steps,
        sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
        sample_1_of_n_eval_on_train_examples=(
            FLAGS.sample_1_of_n_eval_on_train_examples),
        checkpoint_dir=FLAGS.checkpoint_dir,
        wait_interval=60, timeout=FLAGS.eval_timeout)
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
            model_dir=FLAGS.model_dir,
            train_steps=FLAGS.num_train_steps,
            batch_size=FLAGS.batch_size,
            learning_rate=FLAGS.learning_rate,
            fine_tune_checkpoint=FLAGS.fine_tune_checkpoint,
            label_map_path=FLAGS.label_map_path,
            num_classes=FLAGS.num_classes,
            use_tpu=FLAGS.use_tpu,
            save_final_config=True,
            checkpoint_every_n=FLAGS.checkpoint_every_n,
            record_summaries=FLAGS.record_summaries)

if __name__ == '__main__':
#   FLAGS = flags1.FLAGS
#   tf.app.run(create_coco_tf_record())
  FLAGS = flags.FLAGS
  tf2.compat.v1.app.run()
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
import tensorflow.compat.v2 as tf
from object_detection import model_lib_v2
from object_detection.utils import config_util

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

flags.DEFINE_float('learning_rate', 0.001,
                   'learning_rate')
flags.DEFINE_integer('batch_size', 8,
                     'batch_size')
#
flags.DEFINE_string('label_map_path', 'code/object_detection/data/pascal_label_map.pbtxt',
                    'label_map_path')

flags.DEFINE_string('user', 'admin',
                    'user')

flags.DEFINE_string('train_input_path', '/data/dataset/storage/voc/',
                    'train_input_path')

flags.DEFINE_string('eval_input_path', '/data/dataset/storage/voc/',
                    'eval_input_path')

flags.DEFINE_string('fine_tune_checkpoint', 'code/object_detection/data/ssd_efficientdet_d0/ckpt-0',
                    'fine_tune_checkpoint')

flags.DEFINE_string('pipeline_config_path',
                    'code/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_voc12_gpu.config',
                    'Path to pipeline config '
                    'file.')

flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
flags.DEFINE_bool('eval_on_train_data', False, 'Enable evaluating on train '
                                               'data (only supported in distributed training).')
flags.DEFINE_integer('sample_1_of_n_eval_examples', None, 'Will sample one of '
                                                          'every n eval input examples, where n is provided.')
flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample '
                                                                'one of every n train input examples for evaluation, '
                                                                'where n is provided. This is only used if '
                                                                '`eval_training_data` is True.')
flags.DEFINE_string(
    'model_dir', 'work_dirs/ssd_efficientdet_d0', 'Path to output model directory '
                                                  'where event and checkpoint files will be written.')
flags.DEFINE_string(
    'checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
                            '`checkpoint_dir` is provided, this binary operates in eval-only mode, '
                            'writing resulting metrics to `model_dir`.')

flags.DEFINE_integer('eval_timeout', 3600, 'Number of seconds to wait for an'
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
    'checkpoint_every_n', 1000, 'Integer defining how often we checkpoint.')
flags.DEFINE_boolean('record_summaries', True,
                     ('Whether or not to record summaries during'
                      ' training.'))

FLAGS = flags.FLAGS


def transfor_input_path(input_path):
    eval_input_path, train_input_path=None,None
    for roots, dirs, files in os.walk(input_path):
        for f in files:
            path = input_path + '/' + f
            if "train" in f and "record" in f:
                eval_input_path = path
            elif "train" in f and "record" in f:
                train_input_path = path
    return eval_input_path, train_input_path

def main(unused_argv):
    tf.config.set_soft_device_placement(True)
    prefix = os.environ['HOME'] + '/'
    FLAGS.pipeline_config_path = prefix + FLAGS.pipeline_config_path
    FLAGS.label_map_path = prefix + FLAGS.label_map_path
    FLAGS.fine_tune_checkpoint = prefix + FLAGS.fine_tune_checkpoint
    FLAGS.model_dir = prefix + FLAGS.model_dir
    eval_input_path, train_input_path = transfor_input_path(FLAGS.eval_input_path)
    if not  eval_input_path:
        print("该路径下没有tfrecord训练文件")
        return
    if FLAGS.checkpoint_dir:
        FLAGS.checkpoint_dir = prefix + FLAGS.checkpoint_dir
        model_lib_v2.eval_continuously(
            pipeline_config_path=FLAGS.pipeline_config_path,
            model_dir=FLAGS.model_dir,
            train_steps=FLAGS.num_train_steps,
            sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
            sample_1_of_n_eval_on_train_examples=(
                FLAGS.sample_1_of_n_eval_on_train_examples),
            checkpoint_dir=FLAGS.checkpoint_dir,
            wait_interval=300, timeout=FLAGS.eval_timeout,

            learning_rate=FLAGS.learning_rate,
            batch_size=FLAGS.batch_size,
            label_map_path=FLAGS.label_map_path,
            train_input_path=train_input_path,
            eval_input_path=eval_input_path,
        )
    else:
        if FLAGS.use_tpu:
            # TPU is automatically inferred if tpu_name is None and
            # we are running under cloud ai-platform.
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                FLAGS.tpu_name)
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.experimental.TPUStrategy(resolver)
        elif FLAGS.num_workers > 1:
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        else:
            strategy = tf.compat.v2.distribute.MirroredStrategy()

        with strategy.scope():
            model_lib_v2.train_loop(
                pipeline_config_path=FLAGS.pipeline_config_path,
                model_dir=FLAGS.model_dir,
                train_steps=FLAGS.num_train_steps,
                fine_tune_checkpoint=FLAGS.fine_tune_checkpoint,
                use_tpu=FLAGS.use_tpu,
                checkpoint_every_n=FLAGS.checkpoint_every_n,
                record_summaries=FLAGS.record_summaries,

                learning_rate=FLAGS.learning_rate,
                batch_size=FLAGS.batch_size,
                label_map_path=FLAGS.label_map_path,
                train_input_path=train_input_path,
                eval_input_path=eval_input_path,
            )


if __name__ == '__main__':
    tf.compat.v1.app.run()

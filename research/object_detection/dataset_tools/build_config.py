# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
r"""Constructs model, inputs, and training environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import time

from object_detection import model_lib
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import ops

RESTORE_MAP_ERROR_TEMPLATE = (
    'Since we are restoring a v2 style checkpoint'
    ' restore_map was expected to return a (str -> Model) mapping,'
    ' but we received a ({} -> {}) mapping instead.'
)

MODEL_BUILD_UTIL_MAP = model_lib.MODEL_BUILD_UTIL_MAP


def update_configs(pipeline_config_path=None,
                   model_dir=None,
                   config_override=None,
                   train_steps=None,
                   batch_size=None,
                   learning_rate=None,
                   fine_tune_checkpoint=None,
                   label_map_path=None,
                   num_classes=None,
                   train_input_path=None,
                   eval_input_path=None):
    """Trains a model using eager + functions.

    This method:
      1. Processes the pipeline configs
      2. (Optionally) saves the as-run config
      3. Builds the model & optimizer
      4. Gets the training input data
      5. Loads a fine-tuning detection or classification checkpoint if requested
      6. Loops over the train data, executing distributed training steps inside
         tf.functions.
      7. Checkpoints the model every `checkpoint_every_n` training steps.
      8. Logs the training metrics as TensorBoard summaries.
    """

    ## Parse the configs
    get_configs_from_pipeline_file = MODEL_BUILD_UTIL_MAP[
        'get_configs_from_pipeline_file']
    merge_external_params_with_configs = MODEL_BUILD_UTIL_MAP[
        'merge_external_params_with_configs']
    create_pipeline_proto_from_configs = MODEL_BUILD_UTIL_MAP[
        'create_pipeline_proto_from_configs']

    configs = get_configs_from_pipeline_file(
        pipeline_config_path, config_override=None)

    kwargs = {}
    kwargs.update({
        'train_steps': train_steps,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'fine_tune_checkpoint': fine_tune_checkpoint,
        'label_map_path': label_map_path,
        'num_classes': num_classes,
        'train_input_path': train_input_path,
        'eval_input_path': eval_input_path
    })
    configs = merge_external_params_with_configs(
        configs, None, kwargs_dict=kwargs)

    ## update fine_tune_checkpoint
    print("************* force update the fine_tune_checkpoint")
    configs["train_config"].fine_tune_checkpoint = fine_tune_checkpoint
    # Write the as-run pipeline config to disk.
    pipeline_config_final = create_pipeline_proto_from_configs(configs)
    config_util.save_pipeline_config(pipeline_config_final, model_dir)

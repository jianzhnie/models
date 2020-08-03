#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes a ResNetV1-50 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_resnet_v1_50_on_flowers.sh
set -e

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=data/work_dirs/dog-vs-cat-models/resnet_v1_50_from_scrach

# Where the dataset is saved to.
DATASET_DIR=data/dog-vs-cat/

# Download the dataset
python download_and_convert_data.py \
  --dataset_name=dog-vs-cat \
  --dataset_dir=${DATASET_DIR} \
  --num_shards=5

# Fine-tune only the new layers for 3000 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=dog-vs-cat \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v1_50 \
  --max_number_of_steps=100000 \
  --batch_size=32 \
  --learning_rate=0.01 \
  --save_interval_secs=120 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --learning_rate_decay_factor=0.1 \
  --num_epochs_per_decay=200 \
  --weight_decay=0.0001

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=dog-vs-cat \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v1_50


# Fine-tune all the new layers for 1000 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=dog-vs-cat \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --checkpoint_path=${TRAIN_DIR} \
  --model_name=resnet_v1_50 \
  --max_number_of_steps=1000 \
  --batch_size=32 \
  --learning_rate=0.001 \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004


# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=dog-vs-cat \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v1_50

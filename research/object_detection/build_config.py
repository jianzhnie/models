# !/usr/bin/env python
# --------------------------------------------------------
# Copyright (c) DMAI Inc. and its affiliates. All Rights Reserved.
# Licensed under The MIT License [see LICENSE for details]
# Written by zhengwenyong@dm-ai.cn
# --------------------------------------------------------
"""
Building configs for object detection model.
"""
import os
import argparse


def render_config(args):
    placeholder_map = make_placeholder_map(args)
    base_config = get_base_config(args)
    with open(base_config) as f:
        template = f.readlines()

    rendered = []
    for line in template:
        for key, value in placeholder_map.items():
            if line.find(key) != -1:
                line = line.replace(key, str(value))
                break
        rendered.append(line)

    with open(args.output, "w") as f:
        f.writelines(rendered)

    print('save config to "{}"'.format(os.path.abspath(args.output)))


def make_placeholder_map(args):
    base_placeholder_map = {
        "${NUM_CLASSES}": args.num_classes,
        "${NUM_TRAIN_STEPS}": args.num_train_steps,
        "${INITIAL_LEARNING_RATE}": args.learning_rate,
        "${BATCH_SIZE}": args.batch_size,
        "${LABEL_MAP_PATH}": '"' + args.label_map_path + '"',
        "${TRAIN_RECORD}": '"' + args.train_path + '"',
        "${EVAL_RECORD}": '"' + args.val_path + '"',
        "${FINE_TUNE_CHECKPOINT}": '"' + args.fine_tune_checkpoint + '"' if args.fine_tune_checkpoint else '""',
    }

    return base_placeholder_map


def get_base_config(args):
    base_config = "configs/tf2/ssd_efficientdet_d0_512x512_coco17_gpu_base.config"
    return base_config
#!/bin/bash

DATA_DIR=/data/flowers
python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir="${DATA_DIR}"

python download_and_convert_data.py \
    --dataset_name=mnist \
    --dataset_dir=/home/robin/datasets/mnist

python download_and_convert_data.py \
    --dataset_name=hymenoptera \
    --dataset_dir=/home/robin/datasets/hymenoptera_data
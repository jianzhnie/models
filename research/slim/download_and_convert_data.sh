#!/bin/bash

DATA_DIR=/home/robin/datasets/flowers
python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir="${DATA_DIR}"


python download_and_convert_data.py \
    --dataset_name=mnist \
    --dataset_dir=/home/robin/datasets/mnist
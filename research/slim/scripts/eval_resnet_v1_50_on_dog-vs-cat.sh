# Run evaluation.

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=data/work_dirs/dog-vs-cat-models/resnet_v1_50_from_scrach

# Where the dataset is saved to.
DATASET_DIR=data/dog-vs-cat/

python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=dog-vs-cat \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v1_50

# From tensorflow/models/research/
## dataset path
TRAIN_IMAGE_DIR=datasets/coco/train2017/
VAL_IMAGE_DIR=datasets/coco/val2017/
TEST_IMAGE_DIR=datasets/coco/test2017/
TRAIN_ANNOTATIONS_FILE=datasets/coco/annotations/instances_train2017.json
VAL_ANNOTATIONS_FILE=datasets/coco/annotations/instances_val2017.json
TESTDEV_ANNOTATIONS_FILE=datasets/coco/annotations/image_info_test-dev2017.json
OUTPUT_DIR=datasets/coco/

# config path
PIPELINE_CONFIG_PATH=configs/tf2/ssd_efficientdet_d0_512x512_coco17_gpu.config
MODEL_DIR=datasets/work_dirs/coco/ssd_efficientdet_d0_v2
# python model_main_tf2.py \
#     --train_image_dir=${TRAIN_IMAGE_DIR} \
#     --val_image_dir=${VAL_IMAGE_DIR} \
#     --test_image_dir=${TEST_IMAGE_DIR} \
#     --train_annotations_file=${TRAIN_ANNOTATIONS_FILE} \
#     --val_annotations_file=${VAL_ANNOTATIONS_FILE} \
#     --testdev_annotations_file=${TESTDEV_ANNOTATIONS_FILE} \
#     --output_dir=${OUTPUT_DIR} \
#     --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
#     --model_dir=${MODEL_DIR} \
#     --alsologtostderr

python model_main_tf2.py \
    --data_path=datasets/coco \
    --output_path=datasets/work_dirs/coco/ssd_efficientdet_d0_v2 \
    --pipeline_config_path=configs/tf2/ssd_efficientdet_d0_512x512_coco17_gpu.config  \
    --model_dir=datasets/work_dirs/coco/ssd_efficientdet_d0_v2 \
    --label_map_path=data/mscoco_label_map.pbtxt \
    --fine_tune_checkpoint=datasets/work_dirs/checkpoints/efficientnet_b0/ckpt-0 \
    --dataset_name=coco \
    --batch_size=2 \
    --num_train_steps=100 \
    --learning_rate=0.01  \
    --alsologtostderr



CHECKPOINT_DIR=${MODEL_DIR}
python model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --checkpoint_dir=${CHECKPOINT_DIR} \
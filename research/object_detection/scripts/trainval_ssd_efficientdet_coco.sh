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
MODEL_DIR=datasets/work_dirs/coco/ssd_efficientdet_d0

python model_main_tf2.py  --logtostderr \
    --train_image_dir=${TRAIN_IMAGE_DIR} \
    --val_image_dir=${VAL_IMAGE_DIR} \
    --test_image_dir=${TEST_IMAGE_DIR} \
    --train_annotations_file=${TRAIN_ANNOTATIONS_FILE} \
    --val_annotations_file=${VAL_ANNOTATIONS_FILE} \
    --testdev_annotations_file=${TESTDEV_ANNOTATIONS_FILE} \
    --output_dir=${OUTPUT_DIR} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --alsologtostderr

# CHECKPOINT_DIR=${MODEL_DIR}
# python model_main.py \
#     --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
#     --model_dir=${MODEL_DIR} \
#     --checkpoint_dir=${CHECKPOINT_DIR} \
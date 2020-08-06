# From tensorflow/models/research/
TRAIN_IMAGE_DIR=datasets/coco/train2017/
VAL_IMAGE_DIR=datasets/coco/val2017/
TEST_IMAGE_DIR=datasets/coco/test2017/
TRAIN_ANNOTATIONS_FILE=datasets/coco/annotations/instances_train2017.json
VAL_ANNOTATIONS_FILE=datasets/coco/annotations/instances_val2017.json
TESTDEV_ANNOTATIONS_FILE=datasets/coco/annotations/image_info_test-dev2017.json
OUTPUT_DIR=datasets/coco/
python dataset_tools/create_coco_tf_record.py --logtostderr \
    --train_image_dir="${TRAIN_IMAGE_DIR}" \
    --val_image_dir="${VAL_IMAGE_DIR}" \
    --test_image_dir="${TEST_IMAGE_DIR}" \
    --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
    --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
    --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
    --output_dir="${OUTPUT_DIR}"

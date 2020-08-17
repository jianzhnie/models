# From tensorflow/models/research/
## dataset path
OUTPUT_DIR=datasets/coco/

# config path
PIPELINE_CONFIG_PATH=configs/tf2/ssd_efficientdet_d0_512x512_coco17_gpu.config 
MODEL_DIR=datasets/work_dirs/coco/ssd_efficientdet_d0

python model_main_tf2.py --data_path=/data/data_platform/EE7F9151-EBA5-4657-A7F7-17F5772A198C/24782DEE-9507-46CB-A10F-471E0B63EB18 --output_path=datasets/coco/   --pipeline_config_path=configs/tf2/ssd_efficientdet_d0_512x512_coco17_gpu.config  --model_dir=datasets/work_dirs/coco/ssd_efficientdet_d0 --dataset_name="fuck" --alsologtostderr

# CHECKPOINT_DIR=${MODEL_DIR}
# python model_main.py \
#     --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
#     --model_dir=${MODEL_DIR} \
#     --checkpoint_dir=${CHECKPOINT_DIR} \
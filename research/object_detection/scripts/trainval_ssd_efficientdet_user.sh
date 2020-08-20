# From tensorflow/models/research/
## dataset path
OUTPUT_DIR=datasets/coco/

# # config path
# PIPELINE_CONFIG_PATH=configs/tf2/ssd_efficientdet_d0_512x512_coco17_gpu.config 
# MODEL_DIR=datasets/work_dirs/coco/ssd_efficientdet_d0

# python model_main_tf2.py --data_path=/data/data_platform/EE7F9151-EBA5-4657-A7F7-17F5772A198C/24782DEE-9507-46CB-A10F-471E0B63EB18  --dataset_name="fuck" --num_train_steps=100 --alsologtostderr

python model_main_tf2.py --data_path=/data/dataset/storage/kites/ --output_path=/home/admin/work_dirs/ssd_efficientdet_d0/   --pipeline_config_path=configs/tf2/ssd_efficientdet_d0_512x512_coco17_gpu.config   --dataset_name=kites  --checkpoint_path=/home/admin/work_dirs/ssd_efficientdet_d0


# CHECKPOINT_DIR=${MODEL_DIR}
# python model_main.py \
#     --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
#     --model_dir=${MODEL_DIR} \
#     --checkpoint_dir=${CHECKPOINT_DIR} \
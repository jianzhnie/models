# From tensorflow/models/research/
## dataset path
OUTPUT_DIR=datasets/coco/

# # config path
# PIPELINE_CONFIG_PATH=configs/tf2/ssd_efficientdet_d0_512x512_coco17_gpu.config 
# MODEL_DIR=datasets/work_dirs/coco/ssd_efficientdet_d0

# python model_main_tf2.py --data_path=/data/data_platform/EE7F9151-EBA5-4657-A7F7-17F5772A198C/24782DEE-9507-46CB-A10F-471E0B63EB18  --dataset_name="fuck" --num_train_steps=100 --alsologtostderr

python model_main_tf2.py \
    --data_path=datasets/knits \
    --output_path=datasets/work_dirs/coco/ssd_efficientdet_d0_v3 \
    --pipeline_config_path=configs/tf2/ssd_efficientdet_d0_512x512_coco17_gpu.config  \
    --label_map_path=data/mscoco_label_map.pbtxt \
    --fine_tune_checkpoint=datasets/work_dirs/checkpoints/efficientnet_b0/ckpt-0 \
    --dataset_name=knits \
    --batch_size=2 \
    --num_train_steps=200 \
    --learning_rate=0.01  \
    --alsologtostderr



# CHECKPOINT_DIR=${MODEL_DIR}
# python model_main.py \
#     --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
#     --model_dir=${MODEL_DIR} \
#     --checkpoint_dir=${CHECKPOINT_DIR} \
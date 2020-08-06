# From the tensorflow/models/research/ directory
PIPELINE_CONFIG_PATH=configs/tf2/ssd_efficientdet_d0_512x512_voc12_gpu.config 
MODEL_DIR=datasets/work_dirs/ssd_efficientdet_d0 
# python model_main_tf2.py \
#     --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
#     --model_dir=${MODEL_DIR} \
#     --alsologtostderr

CHECKPOINT_DIR=${MODEL_DIR}
python model_main_tf2.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --checkpoint_dir=${CHECKPOINT_DIR} \
    --alsologtostderr
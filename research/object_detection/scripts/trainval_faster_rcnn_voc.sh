# From the tensorflow/models/research/ directory
PIPELINE_CONFIG_PATH=configs/tf2/faster_rcnn_resnet50_v1_800x1333_voc12_gpu.config
MODEL_DIR=datasets/work_dirs/faster_rcnn 
python model_main_tf2.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --alsologtostderr


CHECKPOINT_DIR=${MODEL_DIR}
python model_main_tf2.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --checkpoint_dir=${CHECKPOINT_DIR} \
    --alsologtostderr
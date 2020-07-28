# From the tensorflow/models/research/ directory
python model_main_tf2.py \
    --pipeline_config_path=configs/tf2/faster_rcnn_resnet50_v1_800x1333_voc12_gpu.config \
    --model_dir=datasets/work_dirs/faster_rcnn \
    --alsologtostderr
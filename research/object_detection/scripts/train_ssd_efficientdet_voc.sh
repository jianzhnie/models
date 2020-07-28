# From the tensorflow/models/research/ directory
python model_main_tf2.py \
    --pipeline_config_path=configs/tf2/ssd_efficientdet_d0_512x512_voc12_gpu.config \
    --model_dir=datasets/work_dirs/ssd_efficientdet_d0 \
    --alsologtostderr
# From tensorflow/models/research/

python exporter_main_v2.py  \
    --input_type=[1,512,512,3] \
    --pipeline_config_path=configs/tf2/ssd_efficientdet_d0_512x512_voc12_gpu.config \
    --trained_checkpoint_dir=datasets/work_dirs/ssd_efficientdet_d0/models \
    --output_directory=datasets/work_dirs/ssd_efficientdet_d0/export_ssd_model/


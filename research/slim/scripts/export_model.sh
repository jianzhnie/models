## export_inference_graph
python export_inference_graph.py \
  --alsologtostderr \
  --dataset_name dog-vs-cat \
  --dataset_dir=data/dog-vs-cat \
  --model_name=resnet_v1_50 \
  --output_file=data/work_dirs/dog-vs-cat-models/resnet_v1_50_inf_graph.pb 


## inception_v3
python export_model_graph.py \
  --alsologtostderr \
  --dataset_name  flowers\
  --dataset_dir=data/flowers \
  --model_name=inception_v3 \
  --ckpt_file=data/work_dirs/flowers-models/inception_v3/model.ckpt-1000 \
  --output_dir=data/work_dirs/flowers-models/inception_v3


## vgg
python export_model_graph.py \
  --alsologtostderr \
  --dataset_name  flowers\
  --dataset_dir=data/flowers \
  --model_name=vgg_16 \
  --ckpt_file=data/work_dirs/flowers-models/inception_v3/model.ckpt-1000 \
  --output_dir=data/work_dirs/flowers-models/inception_v3

  
## resnet_v1_50
python export_model_graph.py \
  --alsologtostderr \
  --dataset_name dog-vs-cat \
  --dataset_dir=data/dog-vs-cat \
  --model_name=resnet_v1_50 \
  --ckpt_file=data/work_dirs/dog-vs-cat-models/resnet_v1_50_from_scrach/model.ckpt-1000 \
  --output_dir=data/work_dirs/dog-vs-cat-models/resnet_v1_50_from_scrach 



python exporter_main_v2.py  \
    --input_type=image_tensor \
    --pipeline_config_path=configs/tf2/ssd_efficientdet_d0_512x512_coco17_gpu.config \
    --trained_checkpoint_dir=datasets/work_dirs/coco/ssd_efficientdet_d0 \
    --output_directory=datasets/work_dirs/coco/ssd_efficientdet_d0/export


## custom export 
python export_cls_model_inference_graph.py \
  --alsologtostderr \
  --dataset_name  flowers \
  --dataset_dir=data/flowers \
  --input_type=image_tensor \
  --model_name=inception_v3 \
  --trained_checkpoint_prefix=data/work_dirs/flowers-models/inception_v3/model.ckpt-1000 \
  --output_directory=data/work_dirs/flowers-models/inception_v3/export

python export_cls_model_inference_graph.py \
  --alsologtostderr \
  --dataset_name  dog-vs-cat \
  --dataset_dir=data/dog-vs-cat \
  --input_type=image_tensor \
  --model_name=resnet_v1_50 \
  --trained_checkpoint_prefix=data/work_dirs/dog-vs-cat-models/resnet_v1_50_from_scrach/model.ckpt-1000 \
  --output_directory=data/work_dirs/dog-vs-cat-models/resnet_v1_50_from_scrach/export

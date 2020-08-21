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
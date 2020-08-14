
python export_inference_graph.py \
  --alsologtostderr \
  --dataset_name flowers \
  --dataset_dir=data/flowers \
  --model_name=inception_v3 \
  --output_file=data/work_dirs/inception_v3_inf_graph.pb
# From tensorflow/models/research/

python dataset_tools/create_pascal_tf_record.py \
    --label_map_path=data/pascal_label_map.pbtxt \
    --data_dir=datasets/voc/VOCdevkit --year=VOC2012 --set=train \
    --output_path=datasets/voc/pascal_train.record


python dataset_tools/create_pascal_tf_record.py \
    --label_map_path=data/pascal_label_map.pbtxt \
    --data_dir=datasets/voc/VOCdevkit --year=VOC2012 --set=val \
    --output_path=datasets/voc/pascal_val.record
python client.py --server_url "http://219.133.167.42:30000/endpoints/v2/MzAwNjU=/v1/models/mask:predict" \
--image_path "$(pwd)/object_detection/test_images/image1.jpg" \
--output_json "$(pwd)/object_detection/test_images/out_image1.json" \
--save_output_image "True" \
--label_map "$(pwd)/data/labels.pbtxt"
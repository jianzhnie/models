python client.py --server_url "http://219.133.167.42:30000/endpoints/v2/MzA2NzI=/v1/models/mobilenet:predict" \
--image_path "$(pwd)/object_detection/test_images/image2.jpg" \
--output_json "$(pwd)/object_detection/test_images/out_image2.json" \
--save_output_image "True" \
--label_map "$(pwd)/data/labels.pbtxt"
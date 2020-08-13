python client.py --server_url "http://219.133.167.42:30000/endpoints/v2/MzAzNzc=/v1/models/inference-test:predict" \
--image_path "$(pwd)/img/OIP.jpeg" \
--output_json "$(pwd)/img/out_image2.json" \
--save_output_image "True" \
--label_map "$(pwd)/data/labels.pbtxt"
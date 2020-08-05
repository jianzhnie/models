## Running a serving image

# Download the TensorFlow Serving Docker image and repo
docker pull tensorflow/serving

git clone https://github.com/tensorflow/serving
# Location of demo models
TESTDATA=/home/robin/jianzh/serving/tensorflow_serving/servables/tensorflow/testdata

# Start TensorFlow Serving container and open the REST API port
docker run -t --rm -p 8501:8501 \
    -v "$TESTDATA/saved_model_half_plus_two_cpu:/models/half_plus_two" \
    -e MODEL_NAME=half_plus_two \
    tensorflow/serving &

# Query the model using the predict API
curl -d '{"instances": [1.0, 2.0, 5.0]}' \
    -X POST http://localhost:8501/v1/models/half_plus_two:predict


docker run -p 8500:8500 -p 8501:8501 \
  --mount type=bind,source=/home/robin/datasets/work_dirs/efficientdet/efficientdet_d0_coco17_tpu-32,target=/models/efficientdet \
  --mount type=bind,source=/home/robin/datasets/work_dirs/efficientdet/pipeline.config, target=/models/models.config \
  -t tensorflow/serving:latest --model_config_file=/models/models.config


```sh
MODEL_NAME = ssd_mobilenet_v2_320x320_coco17_tpu-8
MODEL_BASE_PATH = /home/robin/datasets/work_dirs/
tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME}

```

docker run -it tensorflow/serving:latest bash


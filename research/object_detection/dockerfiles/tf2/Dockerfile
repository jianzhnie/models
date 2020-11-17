FROM tensorflow/tensorflow:2.2.0-gpu

ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget \
    vim \
    sudo \


# Install gcloud and gsutil commands
# https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu
RUN export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y

## Install tensorflow models
# RUN git clone https://github.com/jianzhnie/models /models
COPY . /models
# Compile protobuf configs
RUN (cd /models/research/ && protoc object_detection/protos/*.proto --python_out=.)
WORKDIR /models/research/
RUN cp object_detection/packages/tf2/setup.py ./
RUN python -m pip install -U pip
RUN python -m pip install .
# Install jupyter
RUN pip install jupyterlab
ENV TF_CPP_MIN_LOG_LEVEL 3

# TensorFlow Object Detection on Docker

These instructions are experimental.

## Building and running:

```bash
# From the root of the git repository
docker build -f research/object_detection/dockerfiles/tf2/Dockerfile -t tensorflow2.2 .
docker run -it tensorflow
```

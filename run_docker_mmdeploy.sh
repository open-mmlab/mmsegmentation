#!/bin/bash

CONTAINER_NAME="mmdeploy"
IMAGE_TAG=mmdeploy:1.3.0
docker build -t "$IMAGE_TAG" -f deploy_docker/Dockerfile . --progress=plain
docker rm -f "$CONTAINER_NAME"

#  python3 -m tools.torch2onnx 
#  export/as_onnx_opset11.py
#  /data/vit_uper.py
#  /data/work_dir/best_model_name.py
#  /data/dataset/test/images/frame0018.jpg
#  --work-dir=/data/work_dir/

#-d --restart=unless-stopped \
docker run \
  -it \
  --gpus all \
  --shm-size=8g \
  --name "$CONTAINER_NAME" \
  -v "/data:/data" \
  -v "/code/mmsegmentation:/code/mmsegmentation" \
  -w /root/workspace/mmdeploy \
  $IMAGE_TAG \
  bash

#!/bin/bash

CONTAINER_NAME="mmdeploy"
IMAGE_TAG=mmdeploy:1.3.0

BUILD=true
INTERACTIVE=true

function print_usage() {
    printf "Usage: run_docker_mmdeploy.sh [OPTIONS] CMD [ARGS...]

    Example usage:
    - ./run_docker_mmdeploy.sh

    Options: 
        -b: Build image before running              (default: true)
        -i: Run in interactive mode                 (default: true)
        -t set custom image tag                     (default=mmdeploy:1.3.0)
        -h prints this help\n\n"
    if [ ! -z "$1" ]; then
        echo "$@"
        exit 1
    fi  
    exit 0
}

opts="b:i:t:h"
while getopts "$opts" flag; do 
  case "${flag}" in 
    b) BUILD="$OPTARG" ;;
    i) INTERACTIVE="$OPTARG" ;;
    t) IMAGE_TAG="$OPTARG" ;;
    h) print_usage ;;
    *) print_usage "Unrecognized argument '$flag'" ;;
  esac
done
shift $((OPTIND-1))
[[ ! -z $1 ]] || 1=bash

echo "BUILD=$BUILD"
echo "INTERACTIVE=$INTERACTIVE"
echo "IMAGE_TAG=$IMAGE_TAG"

if [ "$BUILD" = true ]; then
    docker build -t "$IMAGE_TAG" -f deploy_docker/Dockerfile . --progress=plain
fi
docker rm -f "$CONTAINER_NAME"

#  python3 -m tools.torch2onnx 
#  export/as_onnx_opset11.py
#  /data/vit_uper.py
#  /data/work_dir/best_model_name.py
#  /data/dataset/test/images/frame0018.jpg
#  --work-dir=/data/work_dir/

if [ "$INTERACTIVE" = true ]; then
    RUN_MODE="-it"
else
    RUN_MODE="-t"
fi

#-d --restart=unless-stopped \
docker run \
  $RUN_MODE \
  --gpus all \
  --shm-size=8g \
  --name "$CONTAINER_NAME" \
  -v "/data:/data" \
  -v "/data/label_studio_datasets:/data/label_studio_datasets" \
  -v "/code/mmsegmentation:/code/mmsegmentation" \
  -w /root/workspace/mmdeploy \
  $IMAGE_TAG \
  "$@"

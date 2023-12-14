#!/bin/bash

CONTAINER_NAME="mmsegmentation"
IMAGE_TAG=mmsegmentation:latest

BUILD=true
INTERACTIVE=true

function print_usage {
    printf "Usage: run_docker_mmseg.sh [OPTIONS] CMD [ARGS...]

    Example usage:
      - ./run_docker_mmseg -r /data/mmsegmentation/experiment_x python3 tools.train configs/amr_segmentation/vit_uper.py  
      - ./run_docker_mmseg -r /data/mmsegmentation/experiment_x bash  

    Options: 
        -d data directory with 'dataset' subfolder    (default=/data/mmsegmentation/)
        -r result directory where to store results to (default=/data/mmsegmentation/model)
        -p pretrain directory with pretrained models  (default=/data/ml_models/models/mmsegmentation/pretrained)
        -b build Docker image before running          (default=true)
        -i run Docker in interactive mode             (default=true)
        -t set custom image tag                       (default=mmsegmentation:latest)
        -h prints this help\n\n"
    if [ ! -z "$1" ]; then
        echo "$@"
        exit 1
    fi  
    exit 0
}

opts="d:p:r:b:i:t:h"
while getopts "$opts" flag; do 
  case "${flag}" in 
    d) DATA_DIR="$OPTARG" ;;
    p) PRETRAIN_DIR="$OPTARG" ;;
    r) RESULT_DIR="$OPTARG" ;;
    b) BUILD="$OPTARG" ;;
    i) INTERACTIVE="$OPTARG" ;;
    t) IMAGE_TAG="$OPTARG" ;;
    h) print_usage ;;
    *) print_usage "Unrecognized argument '$flag'" ;;
  esac
done
shift $((OPTIND-1))
[[ ! -z $1 ]] || 1=bash


REPO_DIR=/code/mmsegmentation
echo "IMAGE_TAG=$IMAGE_TAG"

# defaults and strip tailing slash
DATA_DIR="${DATA_DIR:-/data/mmsegmentation/}"
DATA_DIR="${DATA_DIR%%/}"
echo "DATA_DIR=$DATA_DIR"
RESULT_DIR="${RESULT_DIR:-/data/mmsegmentation/model/}"
RESULT_DIR="${RESULT_DIR%%/}"
echo "RESULT_DIR=$RESULT_DIR"
PRETRAIN_DIR="${PRETRAIN_DIR:-/data/ml_models/models/mmsegmentation/pretrained}"
PRETRAIN_DIR="${PRETRAIN_DIR%%/}"
echo "PRETRAIN_DIR=$PRETRAIN_DIR"


if [ "$BUILD" = true ]; then
    docker build --progress=plain -t $IMAGE_TAG docker/
fi

docker rm -f "$CONTAINER_NAME"

# hack create artificial home for user, with ownership of current host user
mkdir -p "$DATA_DIR/.home"
mkdir -p "$RESULT_DIR"
mkdir -p "$PRETRAIN_DIR"

if [ "$INTERACTIVE" = true ]; then
    RUN_MODE="-it"
else
    RUN_MODE="-t"
fi


#python3 -m tools.train configs/amr_segmentation/vit_uper.py

#-d --restart=unless-stopped \
docker run \
  $RUN_MODE \
  --gpus all \
  --shm-size=8g \
  --name "$CONTAINER_NAME" \
  --user "$(id -u):$(id -g)" \
  -v "/etc/group:/etc/group:ro" \
  -v "/etc/passwd:/etc/passwd:ro" \
  -v "/etc/shadow:/etc/shadow:ro" \
  -v "${DATA_DIR}:/data/" \
  -v "$DATA_DIR/.home:$HOME:rw" \
  -v "${PRETRAIN_DIR}:/mmsegmentation/pretrain/" \
  -v "${RESULT_DIR}:/results/" \
  -v "${REPO_DIR}:/mmsegmentation/" \
  -w /mmsegmentation \
  $IMAGE_TAG \
  "$@"

#!/usr/bin/env bash

# PYTHON="/data/anaconda/envs/pytorch1.5.1/bin/python"
# ./tools/local_train.sh configs/ocrnet/ocrnetplus_r101-d8_512x1024_60k_b16_cityscapes.py 4

PYTHON="/opt/conda/bin/python"
$PYTHON -m pip install -e .

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
${PYTHON} -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch

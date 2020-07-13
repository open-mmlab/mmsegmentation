#!/usr/bin/env bash

PYTHON="/opt/conda/bin/python"
# PYTHON="/data/anaconda/envs/pytorch1.5.1/bin/python"
$PYTHON -m pip install -e .

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
${PYTHON} -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --resume-from $3
    
# ${@:3}
# ./tools/dist_train.sh configs/ocrnet/ocrnet_r101-d8_512x1024_40k_cityscapes.py 8
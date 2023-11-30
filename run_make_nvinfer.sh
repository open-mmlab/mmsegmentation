#!/bin/bash

MODEL_DIR=$1
MODEL_DIR=${MODEL_DIR%%/}
shift

CLASSES=( "$@" )
CLASSES=$(IFS=';' ; echo "${CLASSES[*]}")

echo "[property]

onnx-file=$MODEL_DIR/model.onnx
model-engine-file=$MODEL_DIR/model.onnx_b1_gpu0_fp16.engine

gie-unique-id=4
net-scale-factor=0.00784313725490196
offsets=123.675;116.28;103.53

network-mode=2
batch-size=1

infer-dims=3;512;512
maintain-aspect-ratio=0
model-color-format=0 # 0: RGB # 1: GBR # 2: GRAY

network-type=2 # segmentation 
segmentation-output-order=0 # 0: NCHW # 1: NHWC
#segmentation-threshold=0.0
output-tensor-meta=1

num-detected-classes=2
output-blob-names=output

[custom]
detected-classes=$CLASSES
" > "$MODEL_DIR/nvinfer-segmentation-config.txt"

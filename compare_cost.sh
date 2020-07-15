#!/usr/bin/env bash

PYTHON="/data/anaconda/envs/pytorch1.5.1/bin/python"
# $PYTHON -m pip install -e .

${PYTHON} tools/get_flops.py configs/pspnet/pspnet_r101-d8_512x1024_40k_cityscapes.py
${PYTHON} tools/get_flops.py configs/psanet/psanet_r101-d8_512x1024_40k_cityscapes.py
${PYTHON} tools/get_flops.py configs/nonlocal_net/nonlocal_r101-d8_512x1024_80k_cityscapes.py
${PYTHON} tools/get_flops.py configs/danet/danet_r101-d8_512x1024_40k_cityscapes.py
# ${PYTHON} tools/get_flops.py configs/deeplabv3/deeplabv3_r101-d8_512x1024_40k_cityscapes.py
# ${PYTHON} tools/get_flops.py configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_40k_cityscapes.py


# ${PYTHON} tools/get_flops.py configs/ocrnet/ocrnet_r101-d8_512x1024_40k_cityscapes.py
${PYTHON} tools/get_flops.py configs/ocrnet/ocrnetplus_r101-d8_bs2x_sep_512x1024_40k_cityscapes.py
# ${PYTHON} tools/get_flops.py configs/ocrnet/ocrnetplusv2_r101-d8_bs2x_sep_512x1024_40k_cityscapes.py


# PSPNet: Input shape: (3, 1024, 512) Flops: 512.28 GFLOPs Params: 67.97 M
# PSANet: Input shape: (3, 1024, 512) Flops: 554.86 GFLOPs Params: 78.13 M
# Non-local: Input shape: (3, 1024, 512) Flops: 555.12 GFLOPs Params: 69.02 M
# DANet: Input shape: (3, 1024, 512) Flops: 553.67 GFLOPs Params: 68.84 M
# DeepLabv3: Input shape: (3, 1024, 512) Flops: 694.73 GFLOPs Params: 87.11 M
# DeepLabv3Plus Input shape: (3, 1024, 512) Flops: 508.09 GFLOPs Params: 62.58 M
# OCRNet: Input shape: (3, 1024, 512) Flops: 461.22 GFLOPs Params: 55.52 M
# OCRNetPlusV2: Input shape: (3, 1024, 512) Flops: 411.76 GFLOPs Params: 47.74 M
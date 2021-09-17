_base_ = './pspnet_r50-d8_512x512_4x4_20k_coco-stuff10k.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))

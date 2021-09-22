_base_ = './pspnet_r50-d8_512x512_4x4_80k_coco-stuff164k.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))

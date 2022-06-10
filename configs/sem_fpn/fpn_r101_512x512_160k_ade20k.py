_base_ = './fpn_r50_512x512_160k_ade20k.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
crop_size = (512, 512)
preprocess_cfg = dict(size=crop_size)
model = dict(preprocess_cfg=preprocess_cfg)

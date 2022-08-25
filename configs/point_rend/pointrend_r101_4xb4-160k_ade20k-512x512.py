_base_ = './pointrend_r50_4xb4-160k_ade20k-512x512.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))

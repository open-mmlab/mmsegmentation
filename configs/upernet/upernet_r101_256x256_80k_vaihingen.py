_base_ = './upernet_r50_256x256_80k_vaihingen.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))

_base_ = './pointrend_r50_512x512_160k_rugd.py'
model = dict(pretrained='pretrain/resnet101_v1c-e67eebb6.pth', backbone=dict(depth=101))

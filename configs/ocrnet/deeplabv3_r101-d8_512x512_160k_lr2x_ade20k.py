_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k_lr2x.py'
]
model = dict(
    pretrained='open-mmlab://resnet101_v1c', 
    backbone=dict(depth=101),
    decode_head=dict(num_classes=150), 
    auxiliary_head=dict(num_classes=150),
)
test_cfg = dict(mode='whole')

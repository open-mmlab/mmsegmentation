_base_ = './icnet_r50-d8_4xb2-20k_HOTS_v1-640x480.py'
load_from = "checkpoints/icnet_r50-d8_in1k-pre_832x832_160k_cityscapes_20210926_042715-ce310aea.pth"
model = dict(
    backbone=dict(
        backbone_cfg=dict(
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://resnet50_v1c'))))

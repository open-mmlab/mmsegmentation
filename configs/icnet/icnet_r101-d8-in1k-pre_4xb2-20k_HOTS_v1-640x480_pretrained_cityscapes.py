_base_ = './icnet_r50-d8_4xb2-20k_HOTS_v1-640x480.py'
load_from = "checkpoints/icnet_r101-d8_in1k-pre_832x832_160k_cityscapes_20210925_232612-9484ae8a.pth"
model = dict(
    backbone=dict(
        backbone_cfg=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://resnet101_v1c'))))

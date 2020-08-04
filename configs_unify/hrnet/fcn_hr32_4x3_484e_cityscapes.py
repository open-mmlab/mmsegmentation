_base_ = './fcn_hr18_4x3_484e_cityscapes.py'
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w32',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(32, 64)),
            stage3=dict(num_channels=(32, 64, 128)),
            stage4=dict(num_channels=(32, 64, 128, 256)))),
    decode_head=dict(
        in_channels=[32, 64, 128, 256], channels=sum([32, 64, 128, 256])))

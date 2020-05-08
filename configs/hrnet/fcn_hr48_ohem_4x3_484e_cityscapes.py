_base_ = './fcn_hr18_4x3_484e_cityscapes.py'
model = dict(
    pretrained='pretrain_model/hrnetv2_w48-d2186c55.pth',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        in_channels=[48, 96, 192, 384],
        channels=sum([48, 96, 192, 384]),
        sampler=dict(type='OHEMSegSampler', thresh=0.9, min_kept=131072)))

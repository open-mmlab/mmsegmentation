_base_ = './fcn_hr18_msra_484ep.py'
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w40',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(40, 80)),
            stage3=dict(num_channels=(40, 80, 160)),
            stage4=dict(num_channels=(40, 80, 160, 320)))),
    decode_head=dict(
        in_channels=[40, 80, 160, 320], channels=sum([40, 80, 160, 320])))

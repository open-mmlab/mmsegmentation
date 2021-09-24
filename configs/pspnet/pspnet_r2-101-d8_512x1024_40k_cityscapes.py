_base_ = './pspnet_r101-d8_512x1024_40k_cityscapes.py'
model = dict(
    backbone=dict(
        type='Res2Net',
        depth=101,
        scales=4,
        base_width=26,
        style='pytorch',
        deep_stem=True,
        avg_down=True,
        pretrained='open-mmlab://res2net101_v1d_26w_4s'))

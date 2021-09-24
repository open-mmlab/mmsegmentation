_base_ = './deeplabv3_r101-d8_512x512_80k_ade20k.py'
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

_base_ = './uniformer_fpn_light_xs_1xb2-160k_ade20k-512x512.py'

# model settings
model = dict(
    backbone=dict(
        depth=[2, 5, 8, 2],
        prune_ratio=[[], [], [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                     [0.5, 0.5]],
        trade_off=[[], [], [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5]],
        embed_dim=[56, 112, 224, 448],
        head_dim=28,
        drop_path_rate=0.),
    neck=dict(in_channels=[56, 112, 224, 448]),
)

load_from = '/root/mmsegmentation/fpn_xxs.pth'

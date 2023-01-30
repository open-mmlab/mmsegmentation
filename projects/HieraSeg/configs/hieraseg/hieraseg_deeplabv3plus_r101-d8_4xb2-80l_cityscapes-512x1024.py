_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8_vd_contrast.py',
    '../_base_/datasets/cityscapes.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

custom_imports = dict(imports=[
    'projects.HieraSeg.decode_head.sep_aspp_contrast_head',
    'projects.HieraSeg.losses.hiera_triplet_loss_cityscape'
])

model = dict(
    pretrained=None,
    backbone=dict(depth=101),
    decode_head=dict(
        num_classes=26,
        loss_decode=dict(
            type='HieraTripletLossCityscape', num_classes=19,
            loss_weight=1.0)),
    auxiliary_head=dict(num_classes=19),
    test_cfg=dict(mode='whole', is_hiera=True, hiera_num_classes=7))

_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8_vd_contrast.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    pretrained='https://assets-1257038460.cos.ap-beijing.myqcloud.com/resnet101_v1d.pth', 
    backbone=dict(depth=101),
    decode_head=dict(num_classes=26,
                     loss_decode=dict(type='HieraTripletLossCityscape',
                                      num_classes=19, 
                                      loss_weight=1.0)),
    auxiliary_head=dict(num_classes=19),
    test_cfg=dict(mode='whole', is_hiera=True, hiera_num_classes=7)
)

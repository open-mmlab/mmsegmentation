_base_ = './pspnet_r50-d8_4x4_512x512_80k_potsdam.py'
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))
# model = dict(
#     pretrained=None,
#     backbone=dict(
#         depth=18,
#         init_cfg=dict(
#             type='Pretrained',
#             checkpoint='./pretrained/resnet18_v1c-b5776b93.pth'),
#     ),
#     decode_head=dict(
#         in_channels=512,
#         channels=128,
#         num_classes=6
#     ),
#     auxiliary_head=dict(in_channels=256, channels=64,num_classes=6))

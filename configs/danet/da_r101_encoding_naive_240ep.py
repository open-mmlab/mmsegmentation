_base_ = './da_r50_encoding_240ep.py'
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='pretrain_model/resnet101_encoding-5be5422a.pth',
    backbone=dict(depth=101, norm_cfg=norm_cfg),
    decode_head=dict(norm_cfg=norm_cfg))

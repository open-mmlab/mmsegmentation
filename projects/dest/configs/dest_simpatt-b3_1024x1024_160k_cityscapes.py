_base_ = ['./dest_simpatt-b0_1024x1024_160k_cityscapes.py']

embed_dims = [64, 128, 250, 320]

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=1.)
        }))

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(embed_dims=embed_dims, num_layers=[3, 6, 8, 3]),
    decode_head=dict(in_channels=embed_dims, channels=64))

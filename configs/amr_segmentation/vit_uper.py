_base_ = [
    '../_base_/models/upernet_vit-b16_ln_mln.py',
    '../_base_/datasets/mounted_empty.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_5k.py'
]
model_image_size=(512,512)


optimizer = dict(
  type='AdamW',
  lr=0.00006,
  betas=(0.9, 0.999),
  weight_decay=0.01,
)
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=100,
        end=500,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = dict(batch_size=1)

## MODEL
model = dict(
    data_preprocessor=dict(size=(512,512)),
    pretrained='pretrain/deit_small_patch16_224-cd65a155.pth',
    #pretrained='pretrain/upernet_deit-s16_512x512_160k_ade20k_20210621_160903-5110d916.pth',
    backbone=dict(num_heads=6, embed_dims=384, drop_path_rate=0.1),
    neck=None,
    decode_head=dict(
        num_classes=2,
        out_channels=1,
        loss_decode=dict(use_sigmoid=True),
        in_channels=[384, 384, 384, 384]
    ),
    auxiliary_head=dict(
        num_classes=2, 
        out_channels=1,
        loss_decode=dict(use_sigmoid=True),
        in_channels=384
    ))


_base_ = [
    '../_base_/models/pp_mobile.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
checkpoint = './models/pp_mobile_tiny.pth'
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(data_preprocessor=data_preprocessor,
             backbone=dict(
                 init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
                 type='MobileSeg_Tiny',
                 channels=[16, 24, 32, 64, 128],
                 depths=[2, 2],
                 embed_dims=[64, 128],
                 num_heads=4,
                 inj_type='AAM',
                 out_feat_chs=[32, 64, 128],
             ),
             decode_head=dict(
                 num_classes=150,
                 in_channels=256,
                 use_dw=True,
                 upsample='intepolate'),
             )

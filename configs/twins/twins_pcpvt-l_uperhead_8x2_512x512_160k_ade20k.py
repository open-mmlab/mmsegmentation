_base_ = ['./twins_pcpvt-s_uperhead_8x4_512x512_160k_ade20k.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/twins/pcpvt_large_20220308-37579dc6.pth'  # noqa

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        depths=[3, 8, 27, 3],
        drop_path_rate=0.3))

train_dataloader = dict(batch_size=2, num_workers=2)
val_dataloader = dict(batch_size=2, num_workers=2)
test_dataloader = val_dataloader

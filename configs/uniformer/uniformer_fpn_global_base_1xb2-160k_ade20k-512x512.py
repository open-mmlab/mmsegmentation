_base_ = [
    '../_base_/models/fpn_uniformer.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
# model settings
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(size=crop_size),
    backbone=dict(
        type='UniFormer',
        embed_dim=[64, 128, 320, 512],
        layers=[5, 8, 20, 7],
        head_dim=64,
        drop_path_rate=0.2,
        use_checkpoint=False,
        windows=False,
        hybrid=False),
    neck=dict(in_channels=[64, 128, 320, 512]),
    decode_head=dict(num_classes=150))

train_dataloader = dict(batch_size=2, num_workers=4)
load_from = '/root/mmsegmentation/fpn_global_base.pth'

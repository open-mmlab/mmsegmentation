_base_ = [
    '../_base_/models/fpn_uniformer_light.py',
    '../../../../configs/_base_/datasets/ade20k.py',
    '../../../../configs/_base_/default_runtime.py',
    '../../../../configs/_base_/schedules/schedule_160k.py'
]
custom_imports = dict(imports=['projects.uniformer.backbones'])
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
# model settings
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(size=crop_size),
    backbone=dict(
        type='UniFormer_Light',
        depth=[3, 5, 9, 3],
        conv_stem=True,
        prune_ratio=[[], [], [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                     [0.5, 0.5, 0.5]],
        trade_off=[[], [], [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                   [0.5, 0.5, 0.5]],
        embed_dim=[64, 128, 256, 512],
        head_dim=32,
        mlp_ratio=[3, 3, 3, 3],
        drop_path_rate=0.1),
    neck=dict(in_channels=[64, 128, 256, 512]),
    decode_head=dict(num_classes=150))

train_dataloader = dict(batch_size=2, num_workers=4)
# load_from = '/root/mmsegmentation/fpn_xs.pth'

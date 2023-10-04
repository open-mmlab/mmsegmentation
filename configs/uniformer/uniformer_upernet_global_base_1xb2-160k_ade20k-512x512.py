_base_ = [
    '../_base_/models/upernet_uniformer.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='UniFormer',
        embed_dim=[64, 128, 320, 512],
        layers=[5, 8, 20, 7],
        head_dim=64,
        drop_path_rate=0.4,
        use_checkpoint=True,
        checkpoint_num=[0, 0, 2, 0],
        windows=False,
        hybrid=False),
    decode_head=dict(in_channels=[64, 128, 320, 512], num_classes=150),
    auxiliary_head=dict(in_channels=320, num_classes=150))
train_dataloader = dict(batch_size=2, num_workers=4)
load_from = '/root/mmsegmentation/upernet_global_base.pth'

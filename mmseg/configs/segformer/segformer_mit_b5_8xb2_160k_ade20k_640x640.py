from mmengine.config import read_base

with read_base():
    from .segformer_mit_b0_8xb2_160k_ade20k_512x512 import *
    from .._base_.datasets.ade20k_640x640 import *

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'  # noqa

train_dataloader.update(batch_size=2, num_workers=2)
val_dataloader.update(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

# model settings
crop_size = (640, 640)
data_preprocessor.update(size=crop_size)
model.update(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))

_base_ = [
    '../_base_/models/Adabins.py', '../_base_/datasets/nyu.py',
    '../_base_/default_runtime.py'
]
custom_imports = dict(
    imports=['projects.Adabins.backbones', 'projects.Adabins.decode_head'],
    allow_failed_imports=False)
crop_size = (416, 544)
data_preprocessor = dict(size=crop_size)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(),
    decode_head=dict(),
)

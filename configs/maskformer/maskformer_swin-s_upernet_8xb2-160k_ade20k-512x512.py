_base_ = './maskformer_swin-t_upernet_8xb2-160k_ade20k-512x512.py'

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth'  # noqa
model = dict(
    backbone=dict(depths=[2, 2, 18, 2]),
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file))

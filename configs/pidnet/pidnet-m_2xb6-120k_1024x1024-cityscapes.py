_base_ = './pidnet-s_2xb6-120k_1024x1024-cityscapes.py'
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-m_imagenet1k_20230306-39893c52.pth'  # noqa
model = dict(
    backbone=dict(channels=64, init_cfg=dict(checkpoint=checkpoint_file)),
    decode_head=dict(in_channels=256))

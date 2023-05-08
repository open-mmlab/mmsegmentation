_base_ = './van-b2_pre1k_upernet_4xb2-160k_ade20k-512x512.py'

model = dict(
    backbone=dict(
        depths=[3, 6, 40, 3],
        init_cfg=dict(
            type='Pretrained', checkpoint='pretrained/van_b4_22k.pth'),
        drop_path_rate=0.4))

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=4)

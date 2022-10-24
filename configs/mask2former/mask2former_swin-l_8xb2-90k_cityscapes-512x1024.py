_base_ = ['./mask2former_swin-b_8xb2-90k_cityscapes-512x1024.py']
pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'  # noqa
depths = [2, 2, 6, 2]
model = dict(
    backbone=dict(
        embed_dims=192,
        num_heads=[6, 12, 24, 48],
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    decode_head=dict(num_queries=100, in_channels=[192, 384, 768, 1536]))

train_dataloader = dict(batch_size=1, num_workers=1)

# learning policy
# max_iters = 737500
# param_scheduler = dict(end=max_iters, milestones=[655556, 710184])

# Before 735001th iteration, we do evaluation every 5000 iterations.
# After 735000th iteration, we do evaluation every 737500 iterations,
# which means that we do evaluation at the end of training.'
# interval = 5000
# dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
# train_cfg = dict(
#     max_iters=max_iters,
#     val_interval=interval,
#     dynamic_intervals=dynamic_intervals)

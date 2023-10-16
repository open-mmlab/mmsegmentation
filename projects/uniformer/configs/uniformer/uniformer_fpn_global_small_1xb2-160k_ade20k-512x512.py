_base_ = './uniformer_fpn_global_base_1xb2-160k_ade20k-512x512.py'
# model settings
model = dict(
    backbone=dict(type='UniFormer', layers=[3, 4, 8, 3], drop_path_rate=0.1))

train_dataloader = dict(batch_size=2, num_workers=4)

# load_from = '/root/mmsegmentation/fpn_global_small.pth'

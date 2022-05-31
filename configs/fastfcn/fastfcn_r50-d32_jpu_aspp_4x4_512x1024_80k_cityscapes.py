# model settings
_base_ = './fastfcn_r50-d32_jpu_aspp_512x1024_80k_cityscapes.py'
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=4, num_workers=4)
test_dataloader = val_dataloader

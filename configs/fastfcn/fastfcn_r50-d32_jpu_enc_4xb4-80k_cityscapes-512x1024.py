# model settings
_base_ = './fastfcn_r50-d32_jpu_enc_4xb2-80k_cityscapes-512x1024.py'
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

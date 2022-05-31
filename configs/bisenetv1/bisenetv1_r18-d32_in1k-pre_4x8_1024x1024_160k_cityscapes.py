_base_ = './bisenetv1_r18-d32_in1k-pre_4x4_1024x1024_160k_cityscapes.py'
train_dataloader = dict(batch_size=8, num_workers=4)
val_dataloader = dict(batch_size=8, num_workers=4)
test_dataloader = val_dataloader

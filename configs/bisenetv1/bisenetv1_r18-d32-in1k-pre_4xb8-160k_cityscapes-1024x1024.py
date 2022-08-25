_base_ = './bisenetv1_r18-d32-in1k-pre_4xb4-160k_cityscapes-1024x1024.py'
train_dataloader = dict(batch_size=8, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

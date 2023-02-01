_base_ = ['./setr_vit-l-mla_8xb1-160k_ade20k-512x512.py']

# num_gpus: 8 -> batch_size: 16
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

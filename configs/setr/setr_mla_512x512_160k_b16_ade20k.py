_base_ = ['./setr_mla_512x512_160k_b8_ade20k.py']

# num_gpus: 8 -> batch_size: 16
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=2)
test_dataloader = val_dataloader

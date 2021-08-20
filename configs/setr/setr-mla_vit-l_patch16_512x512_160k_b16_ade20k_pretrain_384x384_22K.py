_base_ = ['./setr-mla_vit-l_patch16_512x512_160k_b8_ade20k_pretrain_384x384_22K.py']

# num_gpus: 8 -> batch_size: 16
data = dict(samples_per_gpu=2)

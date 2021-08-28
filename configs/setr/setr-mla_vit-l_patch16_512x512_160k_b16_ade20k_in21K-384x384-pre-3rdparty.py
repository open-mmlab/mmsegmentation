_base_ = [
    './setr-mla_vit-l_patch16_512x512_160k_b8_ade20k_'
    'in21K-384x384-pre-3rdparty.py'
]

# num_gpus: 8 -> batch_size: 16
data = dict(samples_per_gpu=2)

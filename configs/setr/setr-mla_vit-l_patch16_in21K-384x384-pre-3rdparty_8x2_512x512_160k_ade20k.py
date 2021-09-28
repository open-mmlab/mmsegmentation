_base_ = [
    'setr-mla_vit-l_patch16_in21K-384x384-pre-3rdparty_8x1_'
    '512x512_160k_ade20k.py'
]

# num_gpus: 8 -> batch_size: 16
data = dict(samples_per_gpu=2)

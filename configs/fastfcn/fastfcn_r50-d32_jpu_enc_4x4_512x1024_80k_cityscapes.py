# model settings
_base_ = './fastfcn_r50-d32_jpu_enc_512x1024_80k_cityscapes.py'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)

_base_ = './bisenetv1_r18-d32_in1k-pre_4x4_1024x1024_160k_cityscapes.py'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
)

#configs/deeplabv3/deeplabv3_r50-d8_4xb2-40k_deepglobe-512x1024.py
_base_ = [
    '../_base_/models/deeplabv3_r50-d8_deepGlobe.py', '../_base_/datasets/deepGlobe.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)

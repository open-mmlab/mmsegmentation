_base_ = [
        '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/SARflood.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2))

randomness = dict(
    seed=42,
    deterministic=False,  # 如需完全可复现，设为True
)

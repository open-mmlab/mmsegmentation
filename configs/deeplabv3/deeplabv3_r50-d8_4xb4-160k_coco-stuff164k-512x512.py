_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py',
    '../_base_/datasets/coco-stuff164k.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=171),
    auxiliary_head=dict(num_classes=171))

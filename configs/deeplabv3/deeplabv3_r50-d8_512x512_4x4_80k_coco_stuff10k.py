_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py', '../_base_/datasets/coco_stuff10k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    decode_head=dict(num_classes=171), auxiliary_head=dict(num_classes=171))

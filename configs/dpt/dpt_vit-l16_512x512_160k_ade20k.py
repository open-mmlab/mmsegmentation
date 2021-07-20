_base_ = [
    '../_base_/models/dpt_vit-l16.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

model = dict(
    decode_head=dict(num_classes=150), auxiliary_head=dict(num_classes=150))

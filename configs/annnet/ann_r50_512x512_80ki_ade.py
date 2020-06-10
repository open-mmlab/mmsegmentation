_base_ = [
    '../_base_/models/ann_r50.py', '../_base_/datasets/ade.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80ki.py'
]
model = dict(
    decode_head=dict(num_classes=150), auxiliary_head=dict(num_classes=150))

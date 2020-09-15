_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/pascal_context.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k_lr4e-3.py'
]
model = dict(decode_head=dict(num_classes=60))
test_cfg = dict(mode='slide', crop_size=(480, 480), stride=(320, 320))
_base_ = [
    '../_base_/models/fcn-resize-concat_litehr18-without-head.py',
    '../_base_/datasets/cityscapes.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

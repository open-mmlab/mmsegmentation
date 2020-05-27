_base_ = [
    '../_base_/models/fcn_r50.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40ki.py'
]
model = dict(pretrained='pretrain_model/resnet50_v1c_trick-80a4ced9.pth')

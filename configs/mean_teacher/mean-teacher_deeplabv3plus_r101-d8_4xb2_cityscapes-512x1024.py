_base_ = ['./mean-teacher_deeplabv3plus_r50-d8_4xb2_cityscapes-512x1024.py']

# model settings
deeplabv3plus = dict(
    pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))

model = dict(student=deeplabv3plus, teacher=deeplabv3plus)

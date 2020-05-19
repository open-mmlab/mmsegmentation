_base_ = './deeplabv3_r50_40ki_cityscapes.py'
model = dict(
    pretrained='pretrain_model/resnet101_v1c-5fe8ded3.pth',
    backbone=dict(depth=101),
    decode_head=dict(classes_weight=[
        0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786,
        1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529,
        1.0507
    ]))

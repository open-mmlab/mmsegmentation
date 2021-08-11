_base_ = [
    '../_base_/models/sfnet_r18.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

# img_norm_cfg = dict(
#     mean=[104.00698793, 116.66876762, 122.67891434], std=[1.0, 1.0, 1.0], to_rgb=True)
model = dict(pretrained='torchvision://resnet18',
    test_cfg=dict(mode='slide', crop_size=(864, 864),stride=(576, 576)))




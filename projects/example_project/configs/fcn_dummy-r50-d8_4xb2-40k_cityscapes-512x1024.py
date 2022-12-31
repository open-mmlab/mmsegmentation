_base_ = ['../../../configs/fcn/fcn_r50-d8_4xb2-40k_cityscapes-512x1024.py']

custom_imports = dict(imports=['projects.example_project.dummy'])

crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor, backbone=dict(type='DummyResNet'))

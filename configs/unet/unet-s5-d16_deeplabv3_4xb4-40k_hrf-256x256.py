_base_ = [
    '../_base_/models/deeplabv3_unet_s5-d16.py', '../_base_/datasets/hrf.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(crop_size=(256, 256), stride=(170, 170)))

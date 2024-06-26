_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/hots_v1_640x480.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (640, 480)
data_preprocessor = dict(size=crop_size)
load_from = "checkpoints/fcn_hr18_512x1024_80k_cityscapes_20200601_223255-4e7b345e.pth"
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=46)
    )

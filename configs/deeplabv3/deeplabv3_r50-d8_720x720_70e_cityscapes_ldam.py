_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py',
    '../_base_/datasets/cityscapes_720x720.py', '../_base_/epoch_runtime.py',
    '../_base_/schedules/schedule_70e.py'
]
model = dict(
    decode_head=dict(align_corners=True, num_classes=19, loss_decode=dict(type='LDAMLoss', class_count="./class_count_cityscapes_pixel.npy")),
    # auxiliary_head=dict(align_corners=True),
    # test_cfg=dict(mode='slide', crop_size=(720, 720), stride=(513, 513)),

)
data = dict(samples_per_gpu=4,
            workers_per_gpu=4)

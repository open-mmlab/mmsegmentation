_base_ = [
    '../_base_/models/segmenter_vit-s_mask.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py',
]
find_unused_parameters = True
model = dict(
    backbone=dict(img_size=(512, 512)),
    decode_head=dict(num_classes=150),
    auxiliary_head=[],
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(512, 512)),
)

optimizer = dict(lr=0.001, weight_decay=0.0)

# num_gpus: 8 -> batch_size: 8
data = dict(samples_per_gpu=1, )

# TODO: handle img_norm_cfg

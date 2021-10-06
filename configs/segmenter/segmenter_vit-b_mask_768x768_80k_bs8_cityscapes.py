_base_ = [
    '../_base_/models/segmenter_vit-b_mask.py',
    '../_base_/datasets/cityscapes_769x769.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py',
]
find_unused_parameters = True
model = dict(
    backbone=dict(img_size=(768, 768)),
    decode_head=dict(num_classes=19),
    auxiliary_head=[],
    test_cfg=dict(mode='slide', crop_size=(768, 768), stride=(768, 768)),
)

optimizer = dict(lr=0.01, weight_decay=0.0)

# num_gpus: 8 -> batch_size: 8
data = dict(samples_per_gpu=1, )

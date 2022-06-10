_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    '../_base_/datasets/ade20k.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
preprocess_cfg = dict(size=crop_size)
model = dict(preprocess_cfg=preprocess_cfg)
optimizer = dict(lr=0.001, weight_decay=0.0)

train_dataloader = dict(
    # num_gpus: 8 -> batch_size: 8
    batch_size=1)
val_dataloader = dict(batch_size=1)

_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)

train_dataloader = dict(
    dataset=dict(
        type='BDD100KDataset',
        data_root='data/bdd100k',
        data_prefix=dict(
            img_path='images/10k/train',
            seg_map_path='labels/sem_seg/masks/train'),
    ))
val_dataloader = dict(
    dataset=dict(
        type='BDD100KDataset',
        data_root='data/bdd100k',
        data_prefix=dict(
            img_path='images/10k/val',
            seg_map_path='labels/sem_seg/masks/val'),
    ))

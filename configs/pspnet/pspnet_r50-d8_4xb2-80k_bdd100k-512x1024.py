_base_ = './pspnet_r50-d8_4xb2-80k_cityscapes-512x1024.py'

train_dataloader = dict(
    dataset=dict(
        type='BDD100KDataset',
        data_root='bdd100k',
        data_prefix=dict(
            img_path='images/10k/train',
            seg_map_path='labels/sem_seg/masks/train'),
    ))
val_dataloader = dict(
    dataset=dict(
        type='BDD100KDataset',
        data_root='bdd100k',
        data_prefix=dict(
            img_path='images/10k/val',
            seg_map_path='labels/sem_seg/masks/val'),
    ))
test_dataloader = val_dataloader

# dataset settings
dataset_type = 'PascalContextDataset'
data_root = 'data/VOCdevkit/VOC2010'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClassContext',
        split='ImageSets/SegmentationContext/train.txt'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClassContext',
        split='ImageSets/SegmentationContext/val.txt'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClassContext',
        split='ImageSets/SegmentationContext/val.txt'))
# For fast evaluation during training
evaluation = dict(interval=10, metric='mIoU')

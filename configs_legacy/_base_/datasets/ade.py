# dataset settings
dataset_type = 'ADEDataset'
data_root = 'data/ade/ADEChallengeData2016'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/training',
        ann_dir='annotations/training'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation'))
# For fast evaluation during training
evaluation = dict(interval=10, metric='mIoU')

# dataset settings
dataset_type = 'GID_Dataset'  # 注册的类名
data_root = 'data/gid/'  # 数据集根目录
crop_size = (256, 256)  # 图像裁剪大小
train_pipeline = [
    dict(type='LoadImageFromFile'),  # 从文件中加载图像
    dict(type='LoadAnnotations'),  # 从文件中加载标注
    dict(
        type='RandomResize',  # 随机缩放
        scale=(512, 512),  # 缩放尺寸
        ratio_range=(0.5, 2.0),  # 缩放比例范围
        keep_ratio=True),  # 是否保持长宽比
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),  # 随机裁剪
    dict(type='RandomFlip', prob=0.5),  # 随机翻转
    dict(type='PhotoMetricDistortion'),  # 图像增强
    dict(type='PackSegInputs')  # 打包数据
]
test_pipeline = [
    dict(type='LoadImageFromFile'),  # 从文件中加载图像
    dict(type='Resize', scale=(256, 256), keep_ratio=True),  # 缩放
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),  # 从文件中加载标注
    dict(type='PackSegInputs')  # 打包数据
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]  # 多尺度预测缩放比例
tta_pipeline = [  # 多尺度测试
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(  # 训练数据加载器
    batch_size=2,  # 训练时的数据批量大小
    num_workers=4,  # 数据加载线程数
    persistent_workers=True,  # 是否持久化线程
    sampler=dict(type='InfiniteSampler', shuffle=True),  # 无限采样器
    dataset=dict(
        type=dataset_type,  # 数据集类名
        data_root=data_root,  # 数据集根目录
        data_prefix=dict(
            img_path='img_dir/train',
            seg_map_path='ann_dir/train'),  # 训练集图像和标注路径
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,  # 验证时的数据批量大小
    num_workers=4,  # 数据加载线程数
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

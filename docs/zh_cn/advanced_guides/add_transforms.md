# 自定义数据流程（待更新）

## 数据流程的设计

按照通常的惯例，我们使用 `Dataset` 和 `DataLoader` 做多线程的数据加载。`Dataset` 返回一个数据内容的字典，里面对应于模型前传方法的各个参数。
因为在语义分割中，输入的图像数据具有不同的大小，我们在 MMCV 里引入一个新的 `DataContainer` 类别去帮助收集和分发不同大小的输入数据。

更多细节，请查看[这里](https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/data_container.py) 。

数据的准备流程和数据集是解耦的。通常一个数据集定义了如何处理标注数据（annotations）信息，而一个数据流程定义了准备一个数据字典的所有步骤。一个流程包括了一系列操作，每个操作里都把一个字典作为输入，然后再输出一个新的字典给下一个变换操作。

这些操作可分为数据加载 (data loading)，预处理 (pre-processing)，格式变化 (formatting) 和测试时数据增强 (test-time augmentation)。

下面的例子就是 PSPNet 的一个流程：

```python
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
```

对于每个操作，我们列出它添加、更新、移除的相关字典域 (dict fields)：

### 数据加载 Data loading

`LoadImageFromFile`

- 增加: img, img_shape, ori_shape

`LoadAnnotations`

- 增加: gt_semantic_seg, seg_fields

### 预处理 Pre-processing

`Resize`

- 增加: scale, scale_idx, pad_shape, scale_factor, keep_ratio
- 更新: img, img_shape, \*seg_fields

`RandomFlip`

- 增加: flip
- 更新: img, \*seg_fields

`Pad`

- 增加: pad_fixed_size, pad_size_divisor
- 更新: img, pad_shape, \*seg_fields

`RandomCrop`

- 更新: img, pad_shape, \*seg_fields

`Normalize`

- 增加: img_norm_cfg
- 更新: img

`SegRescale`

- 更新: gt_semantic_seg

`PhotoMetricDistortion`

- 更新: img

### 格式 Formatting

`ToTensor`

- 更新: 由 `keys` 指定

`ImageToTensor`

- 更新: 由 `keys` 指定

`Transpose`

- 更新: 由 `keys` 指定

`ToDataContainer`

- 更新: 由 `keys` 指定

`DefaultFormatBundle`

- 更新: img, gt_semantic_seg

`Collect`

- 增加: img_meta (the keys of img_meta is specified by `meta_keys`)
- 移除: all other keys except for those specified by `keys`

### 测试时数据增强 Test time augmentation

`MultiScaleFlipAug`

## 拓展和使用自定义的流程

1. 在任何一个文件里写一个新的流程，例如 `my_pipeline.py`，它以一个字典作为输入并且输出一个字典

   ```python
   from mmseg.datasets import PIPELINES

   @PIPELINES.register_module()
   class MyTransform:

       def __call__(self, results):
           results['dummy'] = True
           return results
   ```

2. 导入一个新类

   ```python
   from .my_pipeline import MyTransform
   ```

3. 在配置文件里使用它

   ```python
   img_norm_cfg = dict(
       mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
   crop_size = (512, 1024)
   train_pipeline = [
       dict(type='LoadImageFromFile'),
       dict(type='LoadAnnotations'),
       dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
       dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
       dict(type='RandomFlip', flip_ratio=0.5),
       dict(type='PhotoMetricDistortion'),
       dict(type='Normalize', **img_norm_cfg),
       dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
       dict(type='MyTransform'),
       dict(type='DefaultFormatBundle'),
       dict(type='Collect', keys=['img', 'gt_semantic_seg']),
   ]
   ```

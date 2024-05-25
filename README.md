# AI4arctic readme

## Changes done to original mmsegmentation to get it working for ai4arctic

## Template Configuration file:
[configs/multi_task_ai4arctic/mae_ai4arctic_ds5_pt_80_ft_20.py](configs/multi_task_ai4arctic/mae_ai4arctic_ds5_pt_80_ft_20.py)

### Dataset:
- Added a new dataset class `AI4Arctic`
- files added/modified: <br>
[mmseg/datasets/ai4arctic.py](mmseg/datasets/ai4arctic.py)

### Pipelines/image loader:

- Added a new tranform function `PreLoadImageandSegFromNetCDFFile`
- files added/modified: <br>
[mmseg/datasets/transforms/loading_ai4arctic.py](mmseg/datasets/transforms/loading_ai4arctic.py)
- Notes:
-- Loads the GT as 3 channel tensor instead of 1 channel(multiple GT for different tasks)

-- TODO:
1. Upscaling of low res variables
2. Add time, location encoding


### Model (To support multitask):

- Added a new `EncoderDecoder` class called `MultitaskEncoderDecoder` which takes 3 decoder dictionary as input.

eg in config file

```python
decode_head=[
    dict(
        type='UPerHead',
        task='SIC',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=768,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    dict(
        type='UPerHead',
        task='SOD',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=768,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    dict(
        type='UPerHead',
        task='FLOE',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=768,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
],
```

- Modifed class `BaseDecodeHead` to pass the task number, this is required to disitinguish which Ground truth is used to calculate loss.
- [Difference] (https://git.uwaterloo.ca/vip_ai4arctic/mmsegmentation/-/commit/9b4ea6cd9a8a8e93edece0825f71f47f13f0f9d9#669e3eb0aa8bdb6592e42e25e11896ff7c8a2123)
### Metric
- Added a new metric class `MultitaskIoUMetric`
- files added/modified:
[mmseg/evaluation/metrics/multitask_iou_metric.py](mmseg/evaluation/metrics/multitask_iou_metric.py)

### Visualization/ Custom metric Hook

- Added a new hook called `SegAI4ArcticVisualizationHook`
- files added/modified:
[mmseg/engine/hooks/ai4arctic_visualization_hook.py](mmseg/engine/hooks/ai4arctic_visualization_hook.py)
- Notes:
This hook is responsible for calculating R2/F1/Combined score as well as plotting the prediction and saving them to a folder
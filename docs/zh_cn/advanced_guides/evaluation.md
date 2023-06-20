# 模型评测

模型评测过程会分别在 [ValLoop](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L300) 和 [TestLoop](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L373) 中被执行，用户可以在训练期间或使用配置文件中简单设置的测试脚本进行模型性能评估。`ValLoop` 和 `TestLoop` 属于 [Runner](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py#L59)，它们会在第一次被调用时构建。由于 `dataloader` 与 `evaluator` 是必需的参数，所以要成功构建 `ValLoop`，在构建 `Runner` 时必须设置 `val_dataloader` 和 `val_evaluator`，`TestLoop` 亦然。有关 Runner 设计的更多信息，请参阅 [MMEngine](https://github.com/open-mmlab/mmengine) 的[文档](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/design/runner.md)。

<center>
  <img src='../../../resources/test_step.png' />
  <center>测试/验证 数据流</center>
</center>

在 MMSegmentation 中，默认情况下，我们将 dataloader 和 metrics 的设置写在数据集配置文件中，并将 evaluation loop 的配置写在 `schedule_x` 配置文件中。

例如，在 ADE20K 配置文件 `configs/_base_/dataset/ADE20K.py` 中，在第37到48行，我们配置了 `val_dataloader`，在第51行，我们选择 `IoUMetric` 作为 evaluator，并设置 `mIoU` 作为指标：

```python
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
```

为了能够在训练期间进行评估模型，我们将评估配置添加到了 `configs/schedules/schedule_40k.py` 文件的第15至16行：

```python
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
```

使用以上两种设置，MMSegmentation 在 40K 迭代训练期间，每 4000 次迭代进行一次模型 **mIoU** 指标的评估。

如果我们希望在训练后测试模型，则需要将 `test_dataloader`、`test_evaluator` 和 `test_cfg` 配置添加到配置文件中。

```python
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_cfg = dict(type='TestLoop')
```

在 MMSegmentation 中，默认情况下，`test_dataloader` 和 `test_evaluator` 的设置与 `ValLoop` 的 dataloader 和 evaluator 相同，我们可以修改这些设置以满足我们的需要。

## IoUMetric

MMSegmentation 基于 [MMEngine](https://github.com/open-mmlab/mmengine) 提供的 [BaseMetric](https://github.com/open-mmlab/mmengine/blob/main/mmengine/evaluator/metric.py) 实现 [IoUMetric](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/evaluation/metrics/iou_metric.py) 和 [CityscapesMetric](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/evaluation/metrics/citys_metric.py)，以评估模型的性能。有关统一评估接口的更多详细信息，请参阅[文档](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/evaluation.html)。

这里我们简要介绍 `IoUMetric` 的参数和两种主要方法。

除了 `collect_device` 和 `prefix` 之外，`IoUMetric` 的构建还包含一些其他参数。

构造函数的参数：

- ignore_index（int）- 将在评估中忽略的类别索引。默认值：255。
- iou_metrics（list\[str\] | str）- 需要计算的指标，可选项包括 'mIoU'、'mDice' 和 'mFscore'。
- nan_to_num（int，可选）- 如果指定，NaN 值将被用户定义的数字替换。默认值：None。
- beta（int）- 决定综合评分中 recall 的权重。默认值：1。
- collect_device（str）- 用于在分布式训练期间从不同进程收集结果的设备名称。必须是 'cpu' 或 'gpu'。默认为 'cpu'。
- prefix（str，可选）- 将添加到指标名称中的前缀，以消除不同 evaluator 的同名指标的歧义。如果参数中未提供前缀，则将使用 self.default_prefix 进行替代。默认为 None。

`IoUMetric` 实现 IoU 指标的计算，`IoUMetric` 的两个核心方法是 `process` 和 `compute_metrics`。

- `process` 方法处理一批 data 和 data_samples。
- `compute_metrics` 方法根据处理的结果计算指标。

### IoUMetric.process

参数：

- data_batch（Any）- 来自 dataloader 的一批数据。
- data_samples（Sequence\[dict\]）- 模型的一批输出。

返回值：

此方法没有返回值，因为处理的结果将存储在 `self.results` 中，以在处理完所有批次后进行指标的计算。

### IoUMetric.compute_metrics

参数：

- results（list）- 每个批次的处理结果。

返回值：

- Dict\[str，float\] - 计算的指标。指标的名称为 key，值是相应的结果。key 主要包括 **aAcc**、**mIoU**、**mAcc**、**mDice**、**mFscore**、**mPrecision**、**mPrecall**。

## CityscapesMetric

`CityscapesMetric` 使用由 Cityscapes 官方提供的 [CityscapesScripts](https://github.com/mcordts/cityscapesScripts) 进行模型性能的评估。

### 使用方法

在使用之前，请先安装 `cityscapesscripts` 包：

```shell
pip install cityscapesscripts
```

由于 `IoUMetric` 在 MMSegmentation 中作为默认的 evaluator 使用，如果您想使用 `CityscapesMetric`，则需要自定义配置文件。在自定义配置文件中，应按如下方式替换默认 evaluator。

```python
val_evaluator = dict(type='CityscapesMetric', output_dir='tmp')
test_evaluator = val_evaluator
```

### 接口

构造函数的参数：

- output_dir (str) - 预测结果输出的路径
- ignore_index (int) - 将在评估中忽略的类别索引。默认值：255。
- format_only (bool) - 只为提交进行结果格式化而不进行评估。当您希望将结果格式化为特定格式并将其提交给测试服务器时有用。默认为 False。
- keep_results (bool) - 是否保留结果。当 `format_only` 为 True 时，`keep_results` 必须为 True。默认为 False。
- collect_device (str) - 用于在分布式训练期间从不同进程收集结果的设备名称。必须是 'cpu' 或 'gpu'。默认为 'cpu'。
- prefix (str，可选) - 将添加到指标名称中的前缀，以消除不同 evaluator 的同名指标的歧义。如果参数中未提供前缀，则将使用 self.default_prefix 进行替代。默认为 None。

#### CityscapesMetric.process

该方法将在图像上绘制 mask，并将绘制的图像保存到 `work_dir` 中。

参数：

- data_batch（dict）- 来自 dataloader 的一批数据。
- data_samples（Sequence\[dict\]）- 模型的一批输出。

返回值：

此方法没有返回值，因为处理的结果将存储在 `self.results` 中，以在处理完所有批次后进行指标的计算。

#### CityscapesMetric.compute_metrics

此方法将调用 `cityscapessscripts.evaluation.evalPixelLevelSemanticLabeling` 工具来计算指标。

参数：

- results（list）- 数据集的测试结果。

返回值：

- dict\[str:float\] - Cityscapes 评测结果。

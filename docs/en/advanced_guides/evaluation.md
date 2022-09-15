# Evaluation

MMSegmentation implements [IoUMetric](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/evaluation/metrics/iou_metric.py) and [CitysMetric](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/evaluation/metrics/citys_metric.py) for evaluating the performance of models, based on the [BaseMetric](https://github.com/open-mmlab/mmengine/blob/main/mmengine/evaluator/metric.py) provided by [MMEngine](https://github.com/open-mmlab/mmengine). Please refer to [the documentation](https://mmengine.readthedocs.io/en/latest/tutorials/evaluation.html) for more details about the unified evaluation interface.

Users can evaluate model performance during training or using the test script with simple settings in the configuration file.

In MMSegmentation, we write the configuration of metrics in the config files of datasets and the configuration of the evaluation process in the `schedule_x` config files by default.

For example, in the ADE20K config file `configs/_base_/dataset/ade20k.py`, on lines 51 to 52, we select `IoUMetric` as the evaluator and set `mIoU` as the metric:

```python
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
```

To be able to evaluate the model during training, for example, we add the evaluation configuration to the file `configs/schedules/schedule_40k.py` on lines 15 to 17:

```python
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
```

With the above two settings, MMSegmentation evaluates the **mIoU** metric of the model once every 4000 iterations during the training of 40K iterations.

## IoUMetric

Here we briefly describe the arguments and the two main methods of `IoUMetric`.

The constructor of `IoUMetric` has some additional parameters besides the base `collect_device` and `prefix`.

The arguments of the constructor:

- ignore_index (int) - Index that will be ignored in evaluation. Default: 255.
- iou_metrics (list\[str\] | str) - Metrics to be calculated, the options includes 'mIoU', 'mDice' and 'mFscore'.
- nan_to_num (int, optional) - If specified, NaN values will be replaced by the numbers defined by the user. Default: None.
- beta (int) - Determines the weight of recall in the combined score. Default: 1.
- collect_device (str) - Device name used for collecting results from different ranks during distributed training. Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
- prefix (str, optional) - The prefix that will be added in the metric names to disambiguate homonymous metrics of different evaluators. If the prefix is not provided in the argument, self.default_prefix will be used instead. Defaults to None.

`IoUMetric` implements the IoU metric calculation, the core two methods of `IoUMetric` are `process` and `compute_metrics`.

- `process` method processes one batch of data and data_samples.
- `compute_metrics` method computes the metrics from processed results.

#### IoUMetric.process

Parameters:

- data_batch (Any) - A batch of data from the dataloader.
- data_samples (Sequence\[dict\]) - A batch of outputs from the model.

Returns:

This method doesn't have returns since the processed results would be stored in `self.results`, which will be used to compute the metrics when all batches have been processed.

#### IoUMetric.compute_metrics

Parameters:

- results (list) - The processed results of each batch.

Returns:

- Dict\[str, float\] - The computed metrics. The keys are the names of the metrics, and the values are corresponding results. The key mainly includes **aAcc**, **mIoU**, **mAcc**, **mDice**, **mFscore**, **mPrecision**, **mRecall**.

## CitysMetric

`CitysMetric` uses the official [CityscapesScripts](https://github.com/mcordts/cityscapesScripts) provided by Cityscapes to evaluate model performance.

### Usage

Before using it, please install the `cityscapesscripts` package first:

```shell
pip install cityscapesscripts
```

Since the `IoUMetric` is used as the default evaluator in MMSegmentation, if you would like to use `CitysMetric`, customizing the config file is required. In your customized config file, you should overwrite the default evaluator as follows.

```python
val_evaluator = dict(type='CitysMetric', citys_metrics=['cityscapes'])
test_evaluator = val_evaluator
```

### Interface

The arguments of the constructor:

- ignore_index (int) - Index that will be ignored in evaluation. Default: 255.
- citys_metrics (list\[str\] | str) - Metrics to be evaluated, Default: \['cityscapes'\].
- to_label_id (bool) - whether convert output to label_id for submission. Default: True.
- suffix (str): The filename prefix of the png files. If the prefix is "somepath/xxx", the png files will be named "somepath/xxx.png". Default: '.format_cityscapes'.
- collect_device (str): Device name used for collecting results from different ranks during distributed training. Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
- prefix (str, optional): The prefix that will be added in the metric names to disambiguate homonymous metrics of different evaluators. If the prefix is not provided in the argument, self.default_prefix will be used instead. Defaults to None.

#### CitysMetric.process

This method would draw the masks on images and save the painted images to `work_dir`.

Parameters:

- data_batch (Any) - A batch of data from the dataloader.
- data_samples (Sequence\[dict\]) - A batch of outputs from the model.

Returns:

This method doesn't have returns, the annotations' path would be stored in `self.results`, which will be used to compute the metrics when all batches have been processed.

#### CitysMetric.compute_metrics

This method would call `cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling` tool to calculate metrics.

Parameters:

- results (list) - Testing results of the dataset.

Returns:

- dict\[str: float\] - Cityscapes evaluation results.

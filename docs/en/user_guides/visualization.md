# Visualization

MMSegmentation 1.x provides convenient ways for monitoring training status or visualizing data and model predictions.

## Training status Monitor

MMSegmentation 1.x uses TensorBoard to monitor training status.

### TensorBoard Configuration

Install TensorBoard following [official instructions](https://www.tensorflow.org/install) e.g.

```shell
pip install tensorboardX
pip install future tensorboard
```

Add `TensorboardVisBackend` in `vis_backend` of `visualizer` in `default_runtime.py` config file:

```python
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
```

### Examining scalars in TensorBoard

Launch training experiment e.g.

```shell
python tools/train.py configs/pspnet/pspnet_r50-d8_4xb4-80k_ade20k-512x512.py --work-dir work_dir/test_visual
```

Find the `vis_data` path of `work_dir` after starting training, for example, the vis_data path of this particular test is as follows:

```shell
work_dirs/test_visual/20220810_115248/vis_data
```

The scalar file in vis_data path includes learning rate, losses and data_time etc, also record metrics results and you can refer [logging tutorial](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/logging.html) in mmengine to log custom data. The tensorboard visualization results are executed with the following command:

```shell
tensorboard --logdir work_dirs/test_visual/20220810_115248/vis_data
```

## Data and Results visualization

MMSegmentation provides `SegVisualizationHook` that can render segmentation masks of ground truth and prediction. Users can modify `default_hooks` at each `schedule_x.py` config file.

For exsample, In `_base_/schedules/schedule_20k.py`, modify the `SegVisualizationHook` configuration, set `draw` to `True` to enable the storage of network inference results, `interval` indicates the sampling interval of the prediction results, and when set to 1, each inference result of the network will be saved. `interval` is set to 50 by default:

```python
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1))

```

After launch training experiment, visualization results will be stored in the local folder in validation loop,
or when launch evaluation a model on one dataset, the prediction results will be store in the local.
The stored results of the local visualization are kept in `vis_image` under `$WORK_DIRS/vis_data`, e.g.:

```shell
work_dirs/test_visual/20220810_115248/vis_data/vis_image
```

In addition, if `TensorboardVisBackend` is add in `vis_backends`, like [above](#tensorboard-configuration),
we can also run the following command to view them in TensorBoard:

```shell
tensorboard --logdir work_dirs/test_visual/20220810_115248/vis_data
```

If you would like to know more visualization usage, you can refer to [visualization tutorial](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/visualization.html) in mmengie.

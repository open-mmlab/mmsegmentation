# Visualization

MMSegmentation provides segmentation visualization hook, used to visualize validation and testing process prediction results.

## Usage

Users can modify `default_hooks` at each `schedule_x.py` config file.

For exsample, In `_base_/schedules/schedule_20k.py`, modify the `SegVisualizationHook` configuration, set `draw` to `True` to enable the storage of network inference results, `interval` indicates the sampling interval of the prediction results, and when set to 1, each inference result of the network will be saved. `interval` is set to 50 by default:

```python
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1))

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
```

View visualization results in a local folder or use tensorboard.

Find the `vis_data` path of `work_dir` after starting training or testing, for example, the vis_data path of a particular test is as follows:

```shell
work_dirs/test_visual/20220810_115248/vis_data
```

The stored results of the local visualization are kept in `vis_image` under `vis_data`, while the tensorboard visualization results are executed with the following command:

```shell
tensorboard --logdir work_dirs/test_visual/20220810_115248/vis_data
```

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

The scalar file in vis_data path includes learning rate, losses and data_time etc, also record metrics results and you can refer [logging tutorial](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/logging.html) in MMEngine to log custom data. The tensorboard visualization results are executed with the following command:

```shell
tensorboard --logdir work_dirs/test_visual/20220810_115248/vis_data
```

## Data and Results visualization

### Visualizer Data Samples during Model Testing or Validation

MMSegmentation provides `SegVisualizationHook` which is a [hook](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/hook.md) working to visualize ground truth and prediction of segmentation during model testing and evaluation. Its configuration is in `default_hooks`, please see [Runner tutorial](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/runner.md) for more details.

For example, In `_base_/schedules/schedule_20k.py`, modify the `SegVisualizationHook` configuration, set `draw` to `True` to enable the storage of network inference results, `interval` indicates the sampling interval of the prediction results, and when set to 1, each inference result of the network will be saved. `interval` is set to 50 by default:

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

### Visualize a Single Data Sample

If you want to visualize a single data sample, we suggest to use `SegLocalVisualizer`.

`SegLocalVisualizer` is child class inherits from `Visualizer` in MMEngine and works for MMSegmentation visualization, for more details about `Visualizer` please refer to [visualization tutorial](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/visualization.md) in MMEngine.

Here is an example about `SegLocalVisualizer`, first you may download example data below by following commands:

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/189833109-eddad58f-f777-4fc0-b98a-6bd429143b06.png" width="70%"/>
</div>

```shell
wget https://user-images.githubusercontent.com/24582831/189833109-eddad58f-f777-4fc0-b98a-6bd429143b06.png --output-document aachen_000000_000019_leftImg8bit.png
wget https://user-images.githubusercontent.com/24582831/189833143-15f60f8a-4d1e-4cbb-a6e7-5e2233869fac.png --output-document aachen_000000_000019_gtFine_labelTrainIds.png
```

Then you can find their local path and use the scripts below to visualize:

```python
import mmcv
import os.path as osp
import torch
# `PixelData` is data structure for pixel-level annotations or predictions defined in MMEngine.
# Please refer to below tutorial file of data structures in MMEngine:
# https://github.com/open-mmlab/mmengine/tree/main/docs/en/advanced_tutorials/data_element.md

from mmengine.structures import PixelData

# `SegDataSample` is data structure interface between different components
# defined in MMSegmentation, it includes ground truth, prediction and
# predicted logits of semantic segmentation.
# Please refer to below tutorial file of `SegDataSample` for more details:
# https://github.com/open-mmlab/mmsegmentation/blob/1.x/docs/en/advanced_guides/structures.md

from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer

out_file = 'out_file_cityscapes'
save_dir = './work_dirs'

image = mmcv.imread(
    osp.join(
        osp.dirname(__file__),
        './aachen_000000_000019_leftImg8bit.png'
    ),
    'color')
sem_seg = mmcv.imread(
    osp.join(
        osp.dirname(__file__),
        './aachen_000000_000019_gtFine_labelTrainIds.png'  # noqa
    ),
    'unchanged')
sem_seg = torch.from_numpy(sem_seg)
gt_sem_seg_data = dict(data=sem_seg)
gt_sem_seg = PixelData(**gt_sem_seg_data)
data_sample = SegDataSample()
data_sample.gt_sem_seg = gt_sem_seg

seg_local_visualizer = SegLocalVisualizer(
    vis_backends=[dict(type='LocalVisBackend')],
    save_dir=save_dir)

# The meta information of dataset usually includes `classes` for class names and
# `palette` for visualization color of each foreground.
# All class names and palettes are defined in the file:
# https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/utils/class_names.py

seg_local_visualizer.dataset_meta = dict(
    classes=('road', 'sidewalk', 'building', 'wall', 'fence',
             'pole', 'traffic light', 'traffic sign',
             'vegetation', 'terrain', 'sky', 'person', 'rider',
             'car', 'truck', 'bus', 'train', 'motorcycle',
             'bicycle'),
    palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70],
             [102, 102, 156], [190, 153, 153], [153, 153, 153],
             [250, 170, 30], [220, 220, 0], [107, 142, 35],
             [152, 251, 152], [70, 130, 180], [220, 20, 60],
             [255, 0, 0], [0, 0, 142], [0, 0, 70],
             [0, 60, 100], [0, 80, 100], [0, 0, 230],
             [119, 11, 32]])
# When `show=True`, the results would be shown directly,
# else if `show=False`, the results would be saved in local directory folder.
seg_local_visualizer.add_datasample(out_file, image,
                                    data_sample, show=False)
```

Then the visualization result of image with its corresponding ground truth could be found in `./work_dirs/vis_data/vis_image/` whose name is `out_file_cityscapes_0.png`:

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/189835713-c0534054-4bfa-4b75-9254-0afbeb5ff02e.png" width="70%"/>
</div>

If you would like to know more visualization usage, you can refer to [visualization tutorial](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/visualization.html) in MMEngine.

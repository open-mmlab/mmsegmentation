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

### Visualizer Data Samples during Model Testing or Validataion

Noted that `SegVisualizationHook` is used to visualize validation and testing process prediction results, defined [here](../../../mmseg/engine/hooks/visualization_hook.py).

If you want to visualize a single data sample, a pair of input image and its ground truth in the dataset are necessary.
You could prepare them on your own or download examples below by following commands:

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/189833109-eddad58f-f777-4fc0-b98a-6bd429143b06.png" width="70%"/>
</div>

```shell
wget https://user-images.githubusercontent.com/24582831/189833109-eddad58f-f777-4fc0-b98a-6bd429143b06.png --output-document aachen_000000_000019_leftImg8bit.png
wget https://user-images.githubusercontent.com/24582831/189833143-15f60f8a-4d1e-4cbb-a6e7-5e2233869fac.png --output-document aachen_000000_000019_gtFine_labelTrainIds.png
```

Then you can find their local path and use the scrips below to visualize:

```python
import mmcv
import os.path as osp
import torch
from mmengine.structures import PixelData

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

# `PixelData` is data structure for pixel-level annotations or predictions defined in MMEngine.
gt_sem_seg = PixelData(**gt_sem_seg_data)


# `SegDataSample` is data structure interface between different components
# defined in MMSegmentation, it includes ground truth, prediction and
# predicted logits of semantic segmentation.
data_sample = SegDataSample()
data_sample.gt_sem_seg = gt_sem_seg

seg_local_visualizer = SegLocalVisualizer(
    vis_backends=[dict(type='LocalVisBackend')],
    save_dir=save_dir)

# The meta information of dataset usually includes `classes` for class names and
# `palette` for visualization color of each foreground.
# It is usually defined in corresponding class of dataset such as './mmseg/datasets/cityscapes.py'

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

# When `show=False`, the results would be saved in local directory folder.
seg_local_visualizer.add_datasample(out_file, image,
                                    data_sample, show=False)
```

Then the visualization result of image with its corresponding ground truth could be found in `./work_dirs/vis_data/vis_image/` whose name is `out_file_cityscapes_0.png`:

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/189835713-c0534054-4bfa-4b75-9254-0afbeb5ff02e.png" width="70%"/>
</div>

If you would like to know more visualization usage, you can refer to [visualization tutorial](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/visualization.html) in MMEngine.

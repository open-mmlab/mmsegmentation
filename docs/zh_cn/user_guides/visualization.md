# 可视化

MMSegmentation 1.x 提供了简便的方式监控训练时的状态以及可视化在模型预测时的数据。

## 训练状态监控

MMSegmentation 1.x 使用 TensorBoard 来监控训练时候的状态。

### TensorBoard 的配置

安装 TensorBoard 的过程可以按照 [官方安装指南](https://www.tensorflow.org/install) ，具体的步骤如下：

```shell
pip install tensorboardX
pip install future tensorboard
```

在配置文件 `default_runtime.py` 的 `vis_backend` 中添加 `TensorboardVisBackend`。

```python
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
```

### 检查 TensorBoard 中的标量

启动训练实验的命令如下

```shell
python tools/train.py configs/pspnet/pspnet_r50-d8_4xb4-80k_ade20k-512x512.py --work-dir work_dir/test_visual
```

开始训练后找到 `work_dir` 中的 `vis_data` 路径，例如：本次特定测试的 vis_data 路径如下所示：

```shell
work_dirs/test_visual/20220810_115248/vis_data
```

vis_data 路径中的标量文件包括了学习率、损失函数和 data_time 等，还记录了指标结果，您可以参考 MMEngine 中的 [记录日志教程](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/logging.html) 中的日志教程来帮助记录自己定义的数据。 Tensorboard 的可视化结果使用下面的命令执行：

```shell
tensorboard --logdir work_dirs/test_visual/20220810_115248/vis_data
```

## 数据和结果的可视化

### 模型测试或验证期间的可视化数据样本

MMSegmentation 提供了 `SegVisualizationHook` ，它是一个可以用于可视化 ground truth 和在模型测试和验证期间的预测分割结果的[钩子](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/hook.html) 。 它的配置在 `default_hooks` 中，更多详细信息请参见 [执行器教程](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/runner.html)。

例如，在 `_base_/schedules/schedule_20k.py` 中，修改 `SegVisualizationHook` 配置，将 `draw` 设置为 `True` 以启用网络推理结果的存储，`interval` 表示预测结果的采样间隔， 设置为 1 时，将保存网络的每个推理结果。 `interval` 默认设置为 50：

```python
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1))

```

启动训练实验后，可视化结果将在 validation loop 存储到本地文件夹中，或者在一个数据集上启动评估模型时，预测结果将存储在本地。本地的可视化的存储结果保存在 `$WORK_DIRS/vis_data` 下的 `vis_image` 中，例如：

```shell
work_dirs/test_visual/20220810_115248/vis_data/vis_image
```

另外，如果在 `vis_backends` 中添加 `TensorboardVisBackend` ，如 [TensorBoard 的配置](###TensorBoard的配置)，我们还可以运行下面的命令在 TensorBoard 中查看它们：

```shell
tensorboard --logdir work_dirs/test_visual/20220810_115248/vis_data
```

### 可视化单个数据样本

如果你想可视化单个样本数据，我们建议使用 `SegLocalVisualizer` 。

`SegLocalVisualizer`是继承自 MMEngine 中`Visualizer` 类的子类，适用于 MMSegmentation 可视化，有关`Visualizer`的详细信息请参考在 MMEngine 中的[可视化教程](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/visualization.html) 。

以下是一个关于 `SegLocalVisualizer` 的示例，首先你可以使用下面的命令下载这个案例中的数据：

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/189833109-eddad58f-f777-4fc0-b98a-6bd429143b06.png" width="70%"/>
</div>

```shell
wget https://user-images.githubusercontent.com/24582831/189833109-eddad58f-f777-4fc0-b98a-6bd429143b06.png --output-document aachen_000000_000019_leftImg8bit.png
wget https://user-images.githubusercontent.com/24582831/189833143-15f60f8a-4d1e-4cbb-a6e7-5e2233869fac.png --output-document aachen_000000_000019_gtFine_labelTrainIds.png
```

然后你可以找到他们本地的路径和使用下面的脚本文件对其进行可视化：

```python
import mmcv
import os.path as osp
import torch

# `PixelData` 是 MMEngine 中用于定义像素级标注或预测的数据结构。
# 请参考下面的MMEngine数据结构教程文件：
# https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/data_element.html#pixeldata

from mmengine.structures import PixelData

# `SegDataSample` 是在 MMSegmentation 中定义的不同组件之间的数据结构接口，
# 它包括 ground truth、语义分割的预测结果和预测逻辑。
# 详情请参考下面的 `SegDataSample` 教程文件：
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

# 数据集的元信息通常包括类名的 `classes` 和
# 用于可视化每个前景颜色的 `palette` 。
# 所有类名和调色板都在此文件中定义：
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

# 当`show=True`时，直接显示结果，
# 当 `show=False`时，结果将保存在本地文件夹中。

seg_local_visualizer.add_datasample(out_file, image,
                                    data_sample, show=False)
```

可视化后的图像结果和它的对应的 ground truth 图像可以在 `./work_dirs/vis_data/vis_image/` 路径找到，文件名字是：`out_file_cityscapes_0.png` ：

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/189835713-c0534054-4bfa-4b75-9254-0afbeb5ff02e.png" width="70%"/>
</div>

如果你想知道更多的关于可视化的使用指引，你可以参考 MMEngine 中的[可视化教程](<[https://mmengine.readthedocs.io/en/latest/advanced_tutorials/visualization.html](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/advanced_tutorials/visualization.md)>)

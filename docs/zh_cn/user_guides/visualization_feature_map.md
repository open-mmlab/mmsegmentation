# wandb记录特征图可视化

MMSegmentation 1.x 提供了 Weights & Biases 的后端支持，方便对项目代码结果的可视化和管理。

## Wandb的配置

安装 Weights & Biases 的过程可以参考 [官方安装指南](https://docs.wandb.ai/quickstart)，具体的步骤如下:

```shell
pip install wandb
wandb login
```

在 `vis_backend` 中添加 `WandbVisBackend`。

```python
vis_backends=[dict(type='LocalVisBackend'),
              dict(type='TensorboardVisBackend'),
              dict(type='WandbVisBackend')]
```

## 测试数据和结果及特征图的可视化

`SegLocalVisualizer` 是继承自 MMEngine 中 `Visualizer` 类的子类，适用于 MMSegmentation 可视化，有关 `Visualizer` 的详细信息请参考在 MMEngine 中的[可视化教程](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/visualization.html) 。

以下是一个关于 `SegLocalVisualizer` 的示例，首先你可以使用下面的命令下载这个案例中的数据：

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/189833109-eddad58f-f777-4fc0-b98a-6bd429143b06.png" width="70%"/>
</div>

```shell
wget https://user-images.githubusercontent.com/24582831/189833109-eddad58f-f777-4fc0-b98a-6bd429143b06.png --output-document aachen_000000_000019_leftImg8bit.png
wget https://user-images.githubusercontent.com/24582831/189833143-15f60f8a-4d1e-4cbb-a6e7-5e2233869fac.png --output-document aachen_000000_000019_gtFine_labelTrainIds.png

wget https://download.openmmlab.com/mmsegmentation/v0.5/ann/ann_r50-d8_512x1024_40k_cityscapes/ann_r50-d8_512x1024_40k_cityscapes_20200605_095211-049fc292.pth

```

```python
# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from typing import Type

import mmcv
import torch
import torch.nn as nn

from mmengine.model import revert_sync_batchnorm
from mmengine.structures import PixelData
from mmseg.apis import inference_model, init_model
from mmseg.structures import SegDataSample
from mmseg.utils import register_all_modules
from mmseg.visualization import SegLocalVisualizer


class Recorder:
    """record the forward output feature map and save to data_buffer."""

    def __init__(self) -> None:
        self.data_buffer = list()

    def __enter__(self, ):
        self._data_buffer = list()

    def record_data_hook(self, model: nn.Module, input: Type, output: Type):
        self.data_buffer.append(output)

    def __exit__(self, *args, **kwargs):
        pass


def visualize(args, model, recorder, result):
    seg_visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='WandbVisBackend')],
        save_dir='temp_dir',
        alpha=0.5)
    seg_visualizer.dataset_meta = dict(
        classes=model.dataset_meta['classes'],
        palette=model.dataset_meta['palette'])

    image = mmcv.imread(args.img, 'color')

    seg_visualizer.add_datasample(
        name='predict',
        image=image,
        data_sample=result,
        draw_gt=False,
        draw_pred=True,
        wait_time=0,
        out_file=None,
        show=False)

    # add feature map to wandb visualizer
    for i in range(len(recorder.data_buffer)):
        feature = recorder.data_buffer[i][0]  # remove the batch
        drawn_img = seg_visualizer.draw_featmap(
            feature, image, channel_reduction='select_max')
        seg_visualizer.add_image(f'feature_map{i}', drawn_img)

    if args.gt_mask:
        sem_seg = mmcv.imread(args.gt_mask, 'unchanged')
        sem_seg = torch.from_numpy(sem_seg)
        gt_mask = dict(data=sem_seg)
        gt_mask = PixelData(**gt_mask)
        data_sample = SegDataSample()
        data_sample.gt_sem_seg = gt_mask

        seg_visualizer.add_datasample(
            name='gt_mask',
            image=image,
            data_sample=data_sample,
            draw_gt=True,
            draw_pred=False,
            wait_time=0,
            out_file=None,
            show=False)

    seg_visualizer.add_image('image', image)


def main():
    parser = ArgumentParser(
        description='Draw the Feature Map During Inference')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--gt_mask', default=None, help='Path of gt mask file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # show all named module in the model and use it in source list below
    for name, module in model.named_modules():
        print(name)

    source = [
        'decode_head.fusion.stages.0.query_project.activate',
        'decode_head.context.stages.0.key_project.activate',
        'decode_head.context.bottleneck.activate'
    ]
    source = dict.fromkeys(source)

    count = 0
    recorder = Recorder()
    # registry the forward hook
    for name, module in model.named_modules():
        if name in source:
            count += 1
            module.register_forward_hook(recorder.record_data_hook)
            if count == len(source):
                break

    with recorder:
        # test a single image, and record feature map to data_buffer
        result = inference_model(model, args.img)

    visualize(args, model, recorder, result)


if __name__ == '__main__':
    main()

```

将上述代码保存为 feature_map_visual.py，在终端执行如下代码

```shell
python feature_map_visual.py ${图像} ${配置文件} ${检查点文件} [可选参数]
```

样例

```shell
python feature_map_visual.py \
aachen_000000_000019_leftImg8bit.png \
configs/ann/ann_r50-d8_4xb2-40k_cityscapes-512x1024.py \
ann_r50-d8_512x1024_40k_cityscapes_20200605_095211-049fc292.pth \
--gt_mask aachen_000000_000019_gtFine_labelTrainIds.png
```

可视化后的图像结果和它的对应的 feature map图像会出现在wandb账户中

<div align=center>
<img src="https://user-images.githubusercontent.com/76149310/217520321-647f5bf9-eef2-446d-a9e8-5ca7b621d500.png">
</div>

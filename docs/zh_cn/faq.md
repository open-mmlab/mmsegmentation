# 常见问题解答（FAQ）

我们在这里列出了使用时的一些常见问题及其相应的解决方案。 如果您发现有一些问题被遗漏，请随时提 PR 丰富这个列表。 如果您无法在此获得帮助，请使用 [issue模板](https://github.com/open-mmlab/mmsegmentation/blob/master/.github/ISSUE_TEMPLATE/error-report.md/)创建问题，但是请在模板中填写所有必填信息，这有助于我们更快定位问题。

## 安装

兼容的MMSegmentation和MMCV版本如下。请安装正确版本的MMCV以避免安装问题。

| MMSegmentation version |        MMCV version         | MMClassification version |
| :--------------------: | :-------------------------: | :----------------------: |
|         master         |  mmcv-full>=1.5.0, \<1.7.0  | mmcls>=0.20.1, \<=1.0.0  |
|         0.29.0         |  mmcv-full>=1.5.0, \<1.7.0  | mmcls>=0.20.1, \<=1.0.0  |
|         0.28.0         |  mmcv-full>=1.5.0, \<1.7.0  | mmcls>=0.20.1, \<=1.0.0  |
|         0.27.0         |  mmcv-full>=1.5.0, \<1.7.0  | mmcls>=0.20.1, \<=1.0.0  |
|         0.26.0         | mmcv-full>=1.5.0, \<=1.6.0  | mmcls>=0.20.1, \<=1.0.0  |
|         0.25.0         | mmcv-full>=1.5.0, \<=1.6.0  | mmcls>=0.20.1, \<=1.0.0  |
|         0.24.1         | mmcv-full>=1.4.4, \<=1.6.0  | mmcls>=0.20.1, \<=1.0.0  |
|         0.23.0         | mmcv-full>=1.4.4, \<=1.6.0  | mmcls>=0.20.1, \<=1.0.0  |
|         0.22.0         | mmcv-full>=1.4.4, \<=1.6.0  | mmcls>=0.20.1, \<=1.0.0  |
|         0.21.1         | mmcv-full>=1.4.4, \<=1.6.0  |       Not required       |
|         0.20.2         | mmcv-full>=1.3.13, \<=1.6.0 |       Not required       |
|         0.19.0         | mmcv-full>=1.3.13, \<1.3.17 |       Not required       |
|         0.18.0         | mmcv-full>=1.3.13, \<1.3.17 |       Not required       |
|         0.17.0         | mmcv-full>=1.3.7, \<1.3.17  |       Not required       |
|         0.16.0         | mmcv-full>=1.3.7, \<1.3.17  |       Not required       |
|         0.15.0         | mmcv-full>=1.3.7, \<1.3.17  |       Not required       |
|         0.14.1         | mmcv-full>=1.3.7, \<1.3.17  |       Not required       |
|         0.14.0         |  mmcv-full>=1.3.1, \<1.3.2  |       Not required       |
|         0.13.0         |  mmcv-full>=1.3.1, \<1.3.2  |       Not required       |
|         0.12.0         |  mmcv-full>=1.1.4, \<1.3.2  |       Not required       |
|         0.11.0         |  mmcv-full>=1.1.4, \<1.3.0  |       Not required       |
|         0.10.0         |  mmcv-full>=1.1.4, \<1.3.0  |       Not required       |
|         0.9.0          |  mmcv-full>=1.1.4, \<1.3.0  |       Not required       |
|         0.8.0          |  mmcv-full>=1.1.4, \<1.2.0  |       Not required       |
|         0.7.0          |  mmcv-full>=1.1.2, \<1.2.0  |       Not required       |
|         0.6.0          |  mmcv-full>=1.1.2, \<1.2.0  |       Not required       |

如果你安装了mmcv，你需要先运行`pip uninstall mmcv`。
如果mmcv和mmcv-full都安装了，会出现 "ModuleNotFoundError"。

- "No module named 'mmcv.ops'"; "No module named 'mmcv.\_ext'".
  1. 使用`pip uninstall mmcv`卸载环境中现有的mmcv。
  2. 按照[安装说明](get_started#best-practices)安装mmcv-full。

## 如何获知模型训练时需要的显卡数量

- 看模型的config文件的命名。可以参考[学习配置文件](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/zh_cn/tutorials/config.md)中的`配置文件命名风格`部分。比如，对于名字为`segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py`的config文件，`8x1`代表训练其对应的模型需要的卡数为8，每张卡中的batch size为1。
- 看模型的log文件。点开该模型的log文件，并在其中搜索`nGPU`，在`nGPU`后的数字个数即训练时所需的卡数。比如，在log文件中搜索`nGPU`得到`nGPU 0,1,2,3,4,5,6,7`的记录，则说明训练该模型需要使用八张卡。

## auxiliary head 是什么

简单来说，这是一个提高准确率的深度监督技术。在训练阶段，`decode_head` 用于输出语义分割的结果，`auxiliary_head` 只是增加了一个辅助损失，其产生的分割结果对你的模型结果没有影响，仅在在训练中起作用。你可以阅读这篇[论文](https://arxiv.org/pdf/1612.01105.pdf)了解更多信息。

## 为什么日志文件没有被创建

在训练脚本中，我们在第167行调用 `get_root_logger` 方法，然后 mmseg 的 `get_root_logger` 方法调用 mmcv 的 `get_logger`，mmcv 将返回在 'mmsegmentation/tools/train.py' 中使用参数 `log_file` 初始化的同一个 logger。在训练期间只存在一个用 `log_file` 初始化的 logger。

参考：[https://github.com/open-mmlab/mmcv/blob/21bada32560c7ed7b15b017dc763d862789e29a8/mmcv/utils/logging.py#L9-L16](https://github.com/open-mmlab/mmcv/blob/21bada32560c7ed7b15b017dc763d862789e29a8/mmcv/utils/logging.py#L9-L16)

如果你发现日志文件没有被创建，可以检查 `mmcv.utils.get_logger` 是否在其他地方被调用。

## 运行测试脚本时如何输出绘制分割掩膜的图像

在测试脚本中，我们提供了`show-dir`参数来控制是否输出绘制的图像。用户可以运行以下命令:

```shell
python tools/test.py {config} {checkpoint} --show-dir {/path/to/save/image} --opacity 1
```

## 如何处理二值分割任务?

MMSegmentation 使用 `num_classes` 和 `out_channels` 来控制模型最后一层 `self.conv_seg` 的输出. (更多细节可以参考 [这里](https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/decode_heads/decode_head.py).):

```python
def __init__(self,
             ...,
             ):
  ...
  if out_channels is None:
      if num_classes == 2:
          warnings.warn('For binary segmentation, we suggest using'
                        '`out_channels = 1` to define the output'
                        'channels of segmentor, and use `threshold`'
                        'to convert seg_logist into a prediction'
                        'applying a threshold')
      out_channels = num_classes

  if out_channels != num_classes and out_channels != 1:
      raise ValueError(
          'out_channels should be equal to num_classes,'
          'except binary segmentation set out_channels == 1 and'
          f'num_classes == 2, but got out_channels={out_channels}'
          f'and num_classes={num_classes}')

  if out_channels == 1 and threshold is None:
      threshold = 0.3
      warnings.warn('threshold is not defined for binary, and defaults'
                    'to 0.3')
  self.num_classes = num_classes
  self.out_channels = out_channels
  self.threshold = threshold
  ...
  self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)
```

有两种计算二值分割任务的方法. 第一种是当 `out_channels=2` 时, 使用 `F.softmax()` 然后通过 `argmax()` 得到预测结果, 再以 Cross Entropy Loss 作为损失函数. 第二种是当 `out_channels=1` 时, 使用在 [#2016](https://github.com/open-mmlab/mmsegmentation/pull/2016) 提供的参数 `threshold (默认为 0.3)`. 通过 `F.sigmoid()` 和 `threshold` 得到预测结果, 再~~~~以 Binary Cross Entropy Loss 作为损失函数.

```python
...
if self.out_channels == 1:
    seg_logit = F.sigmoid(seg_logit)
else:
    seg_logit = F.softmax(seg_logit, dim=1)

...

if self.out_channels == 1:
    seg_pred = (seg_logit >
                self.decode_head.threshold).to(seg_logit).squeeze(1)
else:
    seg_pred = seg_logit.argmax(dim=1)
```

更多关于计算语义分割预测的细节可以参考 [encoder_decoder.py](https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/segmentors/encoder_decoder.py):

综上所述, 我们建议使用者采取两种解决方案 (1) `num_classes=2`, `out_channels=2` 并在 `CrossEntropyLoss` 里面设置 `use_sigmoid=False` 或者 (2) `num_classes=2`, `out_channels=1` 并在 `CrossEntropyLoss` 里面设置 `use_sigmoid=True`.

当采用解决方案 (2) 时, 下面是对样例 [pspnet_unet_s5-d16.py](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/_base_/models/pspnet_unet_s5-d16.py) 的对应的修改:

```python
decode_head=dict(
    type='PSPHead',
    in_channels=64,
    in_index=4,
    num_classes=2,
    out_channels=1,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
auxiliary_head=dict(
    type='FCNHead',
    in_channels=128,
    in_index=3,
    num_classes=2,
    out_channels=1,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
```

## `reduce_zero_label` 的作用

在 MMSegmentation 里面, 当 [加载注释](https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/datasets/pipelines/loading.py#L91) 时, `reduce_zero_label (bool)` 被用来决定是否将所有 label 减去 1:

```python
if self.reduce_zero_label:
    # avoid using underflow conversion
    gt_semantic_seg[gt_semantic_seg == 0] = 255
    gt_semantic_seg = gt_semantic_seg - 1
    gt_semantic_seg[gt_semantic_seg == 254] = 255
```

`reduce_zero_label` 常常被用来处理 label 0 是背景的数据集, 如果 `reduce_zero_label=True`, label 0 对应的像素将不会参与损失函数的计算. 需要说明的是在二值分割任务中没有必要设置 `reduce_zero_label=True`, 请采用上面我们提到的解决方案.

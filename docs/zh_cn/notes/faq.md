# 常见问题解答（FAQ）

我们在这里列出了使用时的一些常见问题及其相应的解决方案。 如果您发现有一些问题被遗漏，请随时提 PR 丰富这个列表。 如果您无法在此获得帮助，请使用 [issue 模板](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/.github/ISSUE_TEMPLATE/error-report.md/)创建问题，但是请在模板中填写所有必填信息，这有助于我们更快定位问题。

## 安装

兼容的 MMSegmentation 和 MMCV 版本如下。请安装正确版本的 MMCV 以避免安装问题。

| MMSegmentation version |          MMCV version          | MMEngine version  | MMClassification (optional) version | MMDetection (optional) version |
| :--------------------: | :----------------------------: | :---------------: | :---------------------------------: | :----------------------------: |
|     dev-1.x branch     |         mmcv >= 2.0.0          | MMEngine >= 0.7.4 |        mmpretrain>=1.0.0rc7         |         mmdet >= 3.0.0         |
|      main branch       |         mmcv >= 2.0.0          | MMEngine >= 0.7.4 |        mmpretrain>=1.0.0rc7         |         mmdet >= 3.0.0         |
|         1.1.1          |         mmcv >= 2.0.0          | MMEngine >= 0.7.4 |        mmpretrain>=1.0.0rc7         |         mmdet >= 3.0.0         |
|         1.1.0          |         mmcv >= 2.0.0          | MMEngine >= 0.7.4 |        mmpretrain>=1.0.0rc7         |         mmdet >= 3.0.0         |
|         1.0.0          |        mmcv >= 2.0.0rc4        | MMEngine >= 0.7.1 |           mmcls==1.0.0rc6           |         mmdet >= 3.0.0         |
|        1.0.0rc6        |        mmcv >= 2.0.0rc4        | MMEngine >= 0.5.0 |           mmcls>=1.0.0rc0           |       mmdet >= 3.0.0rc6        |
|        1.0.0rc5        |        mmcv >= 2.0.0rc4        | MMEngine >= 0.2.0 |           mmcls>=1.0.0rc0           |        mmdet>=3.0.0rc6         |
|        1.0.0rc4        |        mmcv == 2.0.0rc3        | MMEngine >= 0.1.0 |           mmcls>=1.0.0rc0           |  mmdet>=3.0.0rc4, \<=3.0.0rc5  |
|        1.0.0rc3        |        mmcv == 2.0.0rc3        | MMEngine >= 0.1.0 |           mmcls>=1.0.0rc0           |  mmdet>=3.0.0rc4, \<=3.0.0rc5  |
|        1.0.0rc2        |        mmcv == 2.0.0rc3        | MMEngine >= 0.1.0 |           mmcls>=1.0.0rc0           |  mmdet>=3.0.0rc4, \<=3.0.0rc5  |
|        1.0.0rc1        | mmcv >= 2.0.0rc1, \<=2.0.0rc3> | MMEngine >= 0.1.0 |           mmcls>=1.0.0rc0           |          Not required          |
|        1.0.0rc0        | mmcv >= 2.0.0rc1, \<=2.0.0rc3> | MMEngine >= 0.1.0 |           mmcls>=1.0.0rc0           |          Not required          |

如果您已经安装了版本不合适的 mmcv，请先运行`pip uninstall mmcv`卸载已安装的 mmcv，如您先前安装的为 mmcv-full（存在于 OpenMMLab 1.x），请运行`pip uninstall mmcv-full`进行卸载。

- 如出现 "No module named 'mmcv'"
  1. 使用`pip uninstall mmcv`卸载环境中现有的 mmcv
  2. 按照[安装说明](../get_started.md)安装对应的 mmcv

## 如何获知模型训练时需要的显卡数量

- 看模型的 config 文件命名。可以参考[了解配置文件](../user_guides/1_config.md)中的`配置文件命名风格`部分。比如，对于名字为`segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py`的 config 文件，`8xb1`代表训练其对应的模型需要的卡数为 8，每张卡中的 batch size 为 1。
- 看模型的 log 文件。点开该模型的 log 文件，并在其中搜索`nGPU`，在`nGPU`后的数字个数即训练时所需的卡数。比如，在 log 文件中搜索`nGPU`得到`nGPU 0,1,2,3,4,5,6,7`的记录，则说明训练该模型需要使用八张卡。

## auxiliary head 是什么

简单来说，这是一个提高准确率的深度监督技术。在训练阶段，`decode_head`用于输出语义分割的结果，`auxiliary_head` 只是增加了一个辅助损失，其产生的分割结果对你的模型结果没有影响，仅在在训练中起作用。您可以阅读这篇[论文](https://arxiv.org/pdf/1612.01105.pdf)了解更多信息。

## 运行测试脚本时如何输出绘制分割掩膜的图像

在测试脚本中，我们提供了`--out`参数来控制是否输出保存预测的分割掩膜图像。您可以运行以下命令输出测试结果：

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --out ${OUTPUT_DIR}
```

更多用例细节可查阅[文档](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/docs/zh_cn/user_guides/4_train_test.md#%E6%B5%8B%E8%AF%95%E5%B9%B6%E4%BF%9D%E5%AD%98%E5%88%86%E5%89%B2%E7%BB%93%E6%9E%9C)，[PR #2712](https://github.com/open-mmlab/mmsegmentation/pull/2712) 以及[迁移文档](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/docs/zh_cn/migration/interface.md#%E6%B5%8B%E8%AF%95%E5%90%AF%E5%8A%A8)了解相关说明。

## 如何处理二值分割任务?

MMSegmentation 使用 `num_classes` 和 `out_channels` 来控制模型最后一层 `self.conv_seg` 的输出。更多细节可以参考 [这里](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/mmseg/models/decode_heads/decode_head.py)。

`num_classes` 应该和数据集本身类别个数一致，当是二值分割时，数据集只有前景和背景两类，所以 `num_classes` 为 2. `out_channels` 控制模型最后一层的输出的通道数，通常和 `num_classes` 相等，但当二值分割时候，可以有两种处理方法, 分别是：

- 设置 `out_channels=2`，在训练时以 Cross Entropy Loss 作为损失函数，在推理时使用 `F.softmax()` 归一化 logits 值，然后通过 `argmax()` 得到每个像素的预测结果。

- 设置 `out_channels=1`，在训练时以 Binary Cross Entropy Loss 作为损失函数，在推理时使用 `F.sigmoid()` 和 `threshold` 得到预测结果，`threshold` 默认为 0.3。

对于实现上述两种计算二值分割的方法，需要在 `decode_head` 和 `auxiliary_head` 的配置里修改。下面是对样例 [pspnet_unet_s5-d16.py](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/_base_/models/pspnet_unet_s5-d16.py) 做出的对应修改。

- (1) `num_classes=2`, `out_channels=2` 并在 `CrossEntropyLoss` 里面设置 `use_sigmoid=False`。

```python
decode_head=dict(
    type='PSPHead',
    in_channels=64,
    in_index=4,
    num_classes=2,
    out_channels=2,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
auxiliary_head=dict(
    type='FCNHead',
    in_channels=128,
    in_index=3,
    num_classes=2,
    out_channels=2,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
```

- (2) `num_classes=2`, `out_channels=1` 并在 `CrossEntropyLoss` 里面设置 `use_sigmoid=True`.

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

数据集中 `reduce_zero_label` 参数类型为布尔类型，默认为 False，它的功能是为了忽略数据集 label 0。具体做法是将 label 0 改为 255，其余 label 相应编号减 1，同时 decode head 里将 255 设为 ignore index，即不参与 loss 计算。
以下是 `reduce_zero_label` 具体实现逻辑:

```python
if self.reduce_zero_label:
    # avoid using underflow conversion
    gt_semantic_seg[gt_semantic_seg == 0] = 255
    gt_semantic_seg = gt_semantic_seg - 1
    gt_semantic_seg[gt_semantic_seg == 254] = 255
```

关于您的数据集是否需要使用 reduce_zero_label，有以下两类情况：

- 例如在 [Potsdam](https://github.com/open-mmlab/mmsegmentation/blob/1.x/docs/en/user_guides/2_dataset_prepare.md#isprs-potsdam) 数据集上，有 0-不透水面、1-建筑、2-低矮植被、3-树、4-汽车、5-杂乱，六类。但该数据集提供了两种 RGB 标签，一种为图像边缘处有黑色像素的标签，另一种是没有黑色边缘的标签。对于有黑色边缘的标签，在 [dataset_converters.py](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/tools/dataset_converters/potsdam.py)中，其将黑色边缘转换为 label 0，其余标签分别为 1-不透水面、2-建筑、3-低矮植被、4-树、5-汽车、6-杂乱，那么此时，就应该在数据集 [potsdam.py](https://github.com/open-mmlab/mmsegmentation/blob/ff95416c3b5ce8d62b9289f743531398efce534f/mmseg/datasets/potsdam.py#L23) 中将`reduce_zero_label=True`。如果使用的是没有黑色边缘的标签，那么 mask label 中只有 0-5，此时就应该使`reduce_zero_label=False`。需要结合您的实际情况来使用。
- 例如在第 0 类为 background 类别的数据集上，如果您最终是需要将背景和您的其余类别分开时，是不需要使用`reduce_zero_label`的，此时在数据集中应该将其设置为`reduce_zero_label=False`

**注意:** 使用 `reduce_zero_label` 请确认数据集原始类别个数，如果只有两类，需要关闭 `reduce_zero_label` 即设置 `reduce_zero_label=False`。

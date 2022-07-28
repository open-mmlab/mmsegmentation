# 常见问题解答（FAQ）

我们在这里列出了使用时的一些常见问题及其相应的解决方案。 如果您发现有一些问题被遗漏，请随时提 PR 丰富这个列表。 如果您无法在此获得帮助，请使用 [issue模板](https://github.com/open-mmlab/mmsegmentation/blob/master/.github/ISSUE_TEMPLATE/error-report.md/)创建问题，但是请在模板中填写所有必填信息，这有助于我们更快定位问题。

## 安装

兼容的MMSegmentation和MMCV版本如下。请安装正确版本的MMCV以避免安装问题。

| MMSegmentation version |        MMCV version         | MMClassification version |
| :--------------------: | :-------------------------: | :----------------------: |
|         master         |  mmcv-full>=1.5.0, \<1.7.0  | mmcls>=0.20.1, \<=1.0.0  |
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

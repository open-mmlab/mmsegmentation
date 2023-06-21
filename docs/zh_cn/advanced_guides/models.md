# 模型

我们通常将深度学习任务中的神经网络定义为模型，这个模型即是算法的核心。[MMEngine](https://github.com/open-mmlab/mmengine) 抽象出了一个统一模型 [BaseModel](https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/base_model/base_model.py#L16) 以标准化训练、测试和其他过程。MMSegmentation 实现的所有模型都继承自 `BaseModel`，并且在 MMSegmention 中，我们实现了前向传播并为语义分割算法添加了一些功能。

## 常用组件

### 分割器（Segmentor）

在 MMSegmentation 中，我们将网络架构抽象为**分割器**，它是一个包含网络所有组件的模型。我们已经实现了**编码器解码器（EncoderDecoder）**和**级联编码器解码器（CascadeEncoderDecoder）**，它们通常由**数据预处理器**、**骨干网络**、**解码头**和**辅助头**组成。

### 数据预处理器（Data preprocessor）

**数据预处理器**是将数据复制到目标设备并将数据预处理为模型输入格式的部分。

### 主干网络（Backbone）

**主干网络**是将图像转换为特征图的部分，例如没有最后全连接层的 **ResNet-50**。

### 颈部（Neck）

**颈部**是连接主干网络和头的部分。它对主干网络生成的原始特征图进行一些改进或重新配置。例如 **Feature Pyramid Network（FPN）**。

### 解码头（Decode head）

**解码头**是将特征图转换为分割掩膜的部分，例如 **PSPNet**。

### 辅助头（Auxiliary head）

**辅助头**是一个可选组件，它将特征图转换为仅用于计算辅助损失的分割掩膜。

## 基本接口

MMSegmentation 封装 `BaseModel` 并实现了 [BaseSegmentor](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/models/segmentors/base.py#L15) 类，主要提供 `forward`、`train_step`、`val_step` 和 `test_step` 接口。接下来将详细介绍这些接口。

### forward

<center>
  <img src='https://user-images.githubusercontent.com/15952744/228827860-c0e34875-d370-4736-84f0-9560c26c9576.png' />
  <center>编码器解码器数据流</center>
</center>

<center>
  <center><img src='https://user-images.githubusercontent.com/15952744/228827987-aa214507-0c6d-4a08-8ce4-679b2b200b79.png' /></center>
  <center>级联编码器解码器数据流</center>
</center>

`forward` 方法返回训练、验证、测试和简单推理过程的损失或预测。

该方法应接受三种模式：“tensor”、“predict” 和 “loss”：

- “tensor”：前向推理整个网络并返回张量或张量数组，无需任何后处理，与常见的 `nn.Module` 相同。
- “predict”：前向推理并返回预测值，这些预测值将被完全处理到 `SegDataSample` 列表中。
- “loss”：前向推理并根据给定的输入和数据样本返回损失的`字典`。

**注：**[SegDataSample](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/structures/seg_data_sample.py) 是 MMSegmentation 的数据结构接口，用作不同组件之间的接口。`SegDataSample` 实现了抽象数据元素 `mmengine.structures.BaseDataElement`，请参阅 [MMMEngine](https://github.com/open-mmlab/mmengine) 中的 [SegDataSample 文档](https://mmsegmentation.readthedocs.io/zh_CN/1.x/advanced_guides/structures.html)和[数据元素文档](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/data_element.html)了解更多信息。

注意，此方法不处理在 `train_step` 方法中完成的反向传播或优化器更新。

参数：

- inputs（torch.Tensor）- 通常为形状是（N, C, ...) 的输入张量。
- data_sample（list\[[SegDataSample](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/structures/seg_data_sample.py)\]) - 分割数据样本。它通常包括 `metainfo` 和 `gt_sem_seg` 等信息。默认值为 None。
- mode (str) - 返回什么类型的值。默认为 'tensor'。

返回值：

- `dict` 或 `list`：
  - 如果 `mode == "loss"`，则返回用于反向过程和日志记录的损失张量`字典`。
  - 如果 `mode == "predict"`，则返回 `SegDataSample` 的`列表`，推理结果将被递增地添加到传递给 forward 方法的 `data_sample` 参数中，每个 `SegDataSeample` 包含以下关键词：
    - pred_sm_seg (`PixelData`)：语义分割的预测。
    - seg_logits (`PixelData`)：标准化前语义分割的预测指标。
  - 如果 `mode == "tensor"`，则返回`张量`或`张量数组`的`字典`以供自定义使用。

### 预测模式

我们在[配置文档](../user_guides/1_config.md)中简要描述了模型配置的字段，这里我们详细介绍 `model.test_cfg` 字段。`model.test_cfg` 用于控制前向行为，`"predict"` 模式下的 `forward` 方法可以在两种模式下运行：

- `whole_inference`：如果 `cfg.model.test_cfg.mode == 'whole'`，则模型将使用完整图像进行推理。

  `whole_inference` 模式的一个示例配置：

  ```python
  model = dict(
    type='EncoderDecoder'
    ...
    test_cfg=dict(mode='whole')
  )
  ```

- `slide_inference`：如果 `cfg.model.test_cfg.mode == ‘slide’`，则模型将通过滑动窗口进行推理。**注意：** 如果选择 `slide` 模式，还应指定 `cfg.model.test_cfg.stride` 和 `cfg.model.test_cfg.crop_size`。

  `slide_inference` 模式的一个示例配置：

  ```python
  model = dict(
    type='EncoderDecoder'
    ...
    test_cfg=dict(mode='slide', crop_size=256, stride=170)
  )
  ```

### train_step

`train_step` 方法调用 `loss` 模式的前向接口以获得损失`字典`。`BaseModel` 类实现默认的模型训练过程，包括预处理、模型前向传播、损失计算、优化和反向传播。

参数：

- data (dict or tuple or list) - 从数据集采样的数据。在 MMSegmentation 中，数据字典包含 `inputs` 和 `data_samples` 两个字段。
- optim_wrapper (OptimWrapper) - 用于更新模型参数的 OptimWrapper 实例。

**注：**[OptimWrapper](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/optimizer_wrapper.py#L17) 提供了一个用于更新参数的通用接口，请参阅 [MMMEngine](https://github.com/open-mmlab/mmengine) 中的优化器封装[文档](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/optim_wrapper.html)了解更多信息。

返回值：

-Dict\[str, `torch.Tensor`\]：用于记录日志的张量的`字典`。

<center>
  <img src='https://user-images.githubusercontent.com/15952744/228828089-a9ae1225-958d-4cf7-99af-9af8576f7ef7.png' />
  <center>train_step 数据流</center>
</center>

### val_step

`val_step` 方法调用 `predict` 模式的前向接口并返回预测结果，预测结果将进一步被传递给评测器的进程接口和钩子的 `after_val_inter` 接口。

参数：

- data (`dict` or `tuple` or `list`) - 从数据集中采样的数据。在 MMSegmentation 中，数据字典包含 `inputs` 和 `data_samples` 两个字段。

返回值：

- `list` - 给定数据的预测结果。

<center>
  <img src='https://user-images.githubusercontent.com/15952744/228828179-3269baa3-bebd-4c9a-9787-59e7d785fbcf.png' />
  <center>test_step/val_step 数据流</center>
</center>

### test_step

`BaseModel` 中 `test_step` 与 `val_step` 的实现相同。

## 数据预处理器（Data Preprocessor）

MMSegmentation 实现的 [SegDataPreProcessor](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/models/data_preprocessor.py#L13) 继承自由 [MMEngine](https://github.com/open-mmlab/mmengine) 实现的 [BaseDataPreprocessor](https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/base_model/data_preprocessor.py#L18)，提供数据预处理和将数据复制到目标设备的功能。

Runner 在构建阶段将模型传送到指定的设备，而 [SegDataPreProcessor](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/models/data_preprocessor.py#L13) 在 `train_step`、`val_step` 和 `test_step` 中将数据传送到指定设备，之后处理后的数据将被进一步传递给模型。

`SegDataPreProcessor` 构造函数的参数：

- mean (Sequence\[Number\], 可选) - R、G、B 通道的像素平均值。默认为 None。
- std (Sequence\[Number\], 可选) - R、G、B 通道的像素标准差。默认为 None。
- size (tuple, 可选) - 固定的填充大小。
- size_divisor (int, 可选) - 填充大小的除法因子。
- pad_val (float, 可选) - 填充值。默认值：0。
- seg_pad_val (float, 可选) - 分割图的填充值。默认值：255。
- bgr_to_rgb (bool) - 是否将图像从 BGR 转换为 RGB。默认为 False。
- rgb_to_bgr (bool) - 是否将图像从 RGB 转换为 BGR。默认为 False。
- batch_augments (list\[dict\], 可选) - 批量化的数据增强。默认值为 None。

数据将按如下方式处理：

- 收集数据并将其移动到目标设备。
- 用定义的 `pad_val` 将输入填充到输入大小，并用定义的 `seg_Pad_val` 填充分割图。
- 将输入堆栈到 batch_input。
- 如果输入的形状为 (3, H, W)，则将输入从 BGR 转换为 RGB。
- 使用定义的标准差和平均值标准化图像。
- 在训练期间进行如 Mixup 和 Cutmix 的批量化数据增强。

`forward` 方法的参数：

- data (dict) - 从数据加载器采样的数据。
- training (bool) - 是否启用训练时数据增强。

`forward` 方法的返回值：

- Dict：与模型输入格式相同的数据。

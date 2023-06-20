# 概述

本章节向您介绍 MMSegmentation 框架以及语义分割相关的基本概念。我们还提供了关于 MMSegmentation 的详细教程链接。

## 什么是语义分割？

语义分割是将图像中属于同一目标类别的部分聚类在一起的任务。它也是一种像素级预测任务，因为图像中的每一个像素都将根据类别进行分类。该任务的一些示例基准有 [Cityscapes](https://www.cityscapes-dataset.com/benchmarks/), [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) 和 [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) 。通常用平均交并比 (Mean IoU) 和像素准确率 (Pixel Accuracy) 这两个指标来评估模型。

## 什么是 MMSegmentation?

MMSegmentation 是一个工具箱，它为语义分割任务的统一实现和模型评估提供了一个框架，并且高质量实现了常用的语义分割方法和数据集。

MMSeg 主要包含了 apis, structures, datasets, models, engine, evaluation 和 visualization 这七个主要部分。

- **apis** 提供了模型推理的高级api

- **structures** 提供了分割任务的数据结构 `SegDataSample`

- **datasets** 支持用于语义分割的多种数据集

  - **transforms** 包含多种数据增强变换

- **models** 是分割器最重要的部分，包含了分割器的不同组件

  - **segmentors** 定义了所有分割模型类
  - **data_preprocessors** 用于预处理模型的输入数据
  - **backbones** 包含各种骨干网络，可将图像映射为特征图
  - **necks** 包含各种模型颈部组件，用于连接分割头和骨干网络
  - **decode_heads** 包含各种分割头，将特征图作为输入，并预测分割结果
  - **losses** 包含各种损失函数

- **engine** 是运行时组件的一部分，扩展了 [MMEngine](https://github.com/open-mmlab/mmengine) 的功能

  - **optimizers** 提供了优化器和优化器封装
  - **hooks** 提供了 runner 的各种钩子

- **evaluation** 提供了评估模型性能的不同指标

- **visualization** 分割结果的可视化工具

## 如何使用本指南？

以下是详细步骤，将带您一步步学习如何使用 MMSegmentation :

1. 有关安装说明，请参阅 [开始你的第一步](get_started.md)。

2. 对于初学者来说，MMSegmentation 是开始语义分割之旅的最好选择，因为这里实现了许多 SOTA 模型以及经典的模型 [model](model_zoo.md) 。另外，将各类组件和高级 API 結合使用，可以更便捷的执行分割任务。关于 MMSegmentation 的基本用法，请参考下面的教程：

   - [配置](user_guides/1_config.md)
   - [数据预处理](user_guides/2_dataset_prepare.md)
   - [推理](user_guides/3_inference.md)
   - [训练和测试](user_guides/4_train_test.md)

3. 如果你想了解 MMSegmentation 工作的基本类和功能，请参考下面的教程来深入研究：

   - [数据流](advanced_guides/data_flow.md)
   - [结构](advanced_guides/structures.md)
   - [模型](advanced_guides/models.md)
   - [数据集](advanced_guides/datasets.md)
   - [评估](advanced_guides/evaluation.md)

4. MMSegmentation 也为用户自定义和一些前沿的研究提供了教程，请参考下面的教程来建立你自己的分割项目：

   - [添加新的模型](advanced_guides/add_models.md)
   - [添加新的数据集](advanced_guides/add_datasets.md)
   - [添加新的 transform](advanced_guides/add_transforms.md)
   - [自定义 runtime](advanced_guides/customize_runtime.md)

5. 如果您更熟悉 MMSegmentation v0.x , 以下是 MMSegmentation v0.x 迁移到 v1.x 的文档

   - [迁移](migration/index.rst)

## 参考来源

- https://paperswithcode.com/task/semantic-segmentation/codeless#task-home
